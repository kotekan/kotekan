#ifndef CUDA_GNSS_DESPREAD_KERNEL_HPP
#define CUDA_GNSS_DESPREAD_KERNEL_HPP

#include "../stages/gnss/gnss44.hpp" // the 4+4b codec (CUDA-free, shared with host code)

#include <cuda_runtime.h>
#include <stdint.h>

/**
 * @file
 * @brief GPU GNSS batched despread: fused hop-rate replica generation and complex dot product.
 *
 * A port of gnss::ChannelizedReplicaBank::hoprate_stream (the per-hop channelized replica from
 * precomputed cumulative PFB filter tables) fused with gnss::channelized_despread (data x
 * conj(replica) MAC and replica energy). One launch despreads one record window for a batch of
 * PRN correlator quads over the covering channels.
 *
 * The Phi tables are built on the host per Doppler bucket (GnssCudaDespread, from
 * ChannelizedReplicaBank::hoprate_filter_f32) and rebuilt only when a PRN's Doppler moves by
 * more than refresh_hz; the kernel applies each job's exact code phase and carrier.
 * cuda_gnss_despread_test (cudaGnssDespreadTest.cpp) checks it against the CPU path to ~1e-5
 * relative.
 */
namespace gnss_cuda {

/// One PRN's correlator quad: everything that varies within a batch. Carries its own device Phi
/// table pointers and filter span, so one launch can mix PRNs from different Doppler buckets.
///
/// One job emits four output rows (E, P, L, P_HEAD); see launch_despread. Early and Late are the
/// prompt displaced by +-ds, and all four share one carrier (same wc) and one data sample per
/// hop, so the kernel generates the carrier and loads the data once per (PRN, channel, hop).
/// P_HEAD is the prompt's own MAC, gated on the hop.
struct DespreadJob {
    double cp0; ///< prompt code phase, combined-stream chips, referred to absolute sample 0
                ///< (host-side; the kernels use cp_ref)
    double ds;  ///< Early/Late displacement from the prompt (combined-stream chips):
                ///< row 0 = prompt-ds, row 1 = prompt, row 2 = prompt+ds
    /// Prompt code phase at the window's reference sample p.n0, combined-stream chips, reduced
    /// mod code_len on the host in long double.
    ///
    /// Evaluated in double on the absolute sample, cp0 + n*cps reaches ~6e12 chips at
    /// n ~ 1.9e15 (about a week of F-engine uptime), where a double's ulp is ~1e-3 chips, and
    /// the rounding changes from hop to hop within a record, so no per-record correction can
    /// remove it. From cp_ref the kernel multiplies cps only by the intra-record offset.
    ///
    /// n0 is the hop's last sample, window_start + fft_len - 1, as for ang0 and
    /// GnssCudaDespread's m_head_for.
    double cp_ref;
    double cps;     ///< chips per sample incl. code Doppler: eff_chip_rate/fs*(1+sign*f/f_c)
    double inv_cps; ///< 1/cps (tap-boundary indices via multiply, not the costly FP64 divide)
    double wc;      ///< carrier angular rate, rad/sample: 2*pi*(carrier offset + Doppler +
                    ///< carrier trim)/fs
    /// Carrier phase at the window's reference sample p.n0, radians in [0, 2*pi), computed on
    /// the host in long double.
    ///
    /// The kernel evaluates ang0 + wc*(n - n0), where n - n0 stays within one record
    /// (<= 3.4e7 samples), so the rounding of wc (the ulp of a ~1.18 GHz carrier sum is
    /// 2.4e-7 Hz) contributes ~1.5e-8 rad. Evaluated as wc*n on the absolute sample
    /// (n ~ 1.9e15 after a week), the same rounding gives tenths of a radian. That error is
    /// constant within a record and changes every record as the Doppler is re-propagated, and
    /// per-record constants are what the cross-record estimators integrate.
    ///
    /// n0 must be the hop's last sample (window_start + fft_len - 1), matching p.n0 and
    /// m_head_for; any other n0 rotates the whole record by a constant.
    double ang0;
    int code_offset;    ///< this PRN's offset into the shared code table
    int code_len;       ///< combined-stream code length (chips)
    uint64_t chan_mask; ///< bit ci set = channel ci is in this PRN's covering set (<=64 chans)
    /// [n_chan][Lf+1] cumulative filter tables for this PRN's Doppler bucket, or the shared
    /// Doppler-free tables when DespreadParams::shared is set: float2, or __half2 storage when
    /// DespreadParams::phi_half is set.
    const float2* phiA;
    const float2* phiB;
    int n_chips; ///< chips spanned by this bucket's filter; the gather walks [d_first, n_chips)
    int m_head;  ///< row 3 (P_HEAD) is the prompt over hops [0, m_head) only: the part of the
                 ///< window before the code-period boundary, where the secondary overlay or nav
                 ///< symbol can flip sign (gnssRecord.hpp REC_PH_*). 0 gives an all-zero row 3
                 ///< (despread_batch's plain E/P/L contract).
    /// Shared-table mode: the first-moment companion tables Psi and this PRN's carrier offset
    /// from the tables' own carrier, rad/sample (see
    /// ChannelizedReplicaBank::HopRateFilter::PsiA). Read only when DespreadParams::shared is
    /// set; otherwise null and 0. These and d_first come last because the jobs are built with
    /// positional aggregate initializers (GnssCudaDespread.cpp, cudaGnssDespreadTest.cpp), and a
    /// field inserted earlier would shift every later initializer.
    const float2* psiA = nullptr;
    const float2* psiB = nullptr;
    float ddw = 0.0f;
    /// First chip of the gather walk, for centered truncation; 0 walks the span from chip 0.
    /// With n_chips capped, (d_first, n_chips) selects the central window of the PFB span,
    /// where the prototype peaks. Set from the same tables as n_chips.
    int d_first = 0;
};

/// One PRN's peel: its prompt replica, times a per-channel gain, is subtracted from the voltage.
///
/// A @ref DespreadJob without @c ds (Early and Late are correlation taps, with no signal of
/// their own to remove) plus the gains. The peel and the despread must generate bit-identical
/// replicas, or the analytic add-back is not exact.
///
/// The gains are feed-forward: from previous records' tracked gain, not a fit to this window.
/// That keeps the peel to one pass, makes the add-back exact for any gain, and keeps the
/// residual an honest depth metric (a gain fitted in-window subtracts its own noise along R, and
/// the residual reads ~0).
///
/// There are two gains because the window can straddle a code-period boundary at hop
/// @c m_head, where the overlay or nav symbol can flip sign; one gain would model a sign flip
/// with a constant and remove almost nothing. @c a_head applies to hops [0, m_head), @c a_tail to
/// [m_head, n_hops): the same boundary as the despread's P_HEAD row.
struct PeelJob {
    double cp0; ///< prompt code phase, combined-stream chips, referred to absolute sample 0
                ///< (host-side; the kernel uses cp_ref)
    /// Prompt code phase at p.n0 (see DespreadJob::cp_ref). Must be built by the same
    /// expression as the despread's, or the replicas differ.
    double cp_ref;
    double cps;     ///< chips per sample incl. code Doppler (== DespreadJob::cps)
    double inv_cps; ///< 1/cps
    double wc;      ///< carrier angular rate, as DespreadJob::wc but without the carrier trim
    /// Carrier phase at p.n0 (see DespreadJob::ang0), without the carrier trim, as for wc.
    /// Otherwise it must be built by the same expression as the despread's, or the replicas
    /// differ.
    double ang0;
    int code_offset;    ///< offset into the shared code table
    int code_len;       ///< combined-stream code length (chips)
    uint64_t chan_mask; ///< bit ci set = channel ci is in this PRN's covering set
    const float2* phiA; ///< cumulative filter tables, this PRN's Doppler bucket (as DespreadJob)
    const float2* phiB;
    int n_chips;          ///< chips spanned by this bucket's filter
    int m_head;           ///< code-period boundary hop: a_head below it, a_tail at and above it
                          ///< (n_hops: a_head throughout; 0: a_tail throughout)
    const float2* a_head; ///< [n_chan] gain to subtract, hops [0, m_head)
    const float2* a_tail; ///< [n_chan] gain to subtract, hops [m_head, n_hops)
    /// Set as in DespreadJob, but gnss_peel_kernel does not read them: it gathers from phiA/phiB
    /// with no Psi correction, so with shared tables the peel's replica is not the despread's.
    /// Last in the struct, as in DespreadJob.
    const float2* psiA = nullptr;
    const float2* psiB = nullptr;
    float ddw = 0.0f;
    /// As DespreadJob::d_first. Must match the despread's truncation, or the replicas differ.
    int d_first = 0;
};

/// Batch-shared geometry.
struct DespreadParams {
    long long n0; ///< reference sample of hop 0, window_start + fft_len - 1 (the hop's last
                  ///< sample); the host evaluates cp_ref and ang0 here
    int fft_len;  ///< samples per hop
    int n_hops;   ///< hops per record
    int Lf;       ///< filter length fft_len*num_taps (Phi tables have Lf+1 entries)
    /// Output rows per spec in corr/energy: 4 (gnss_gpu::ROWS_PLAIN), or 6 (ROWS_PEEL) when the
    /// frame also carries the peel residual rows, so the despread writes E/P/L/P_HEAD at the
    /// frame's stride. xcorr keeps a stride of 4; it is scratch, not frame.
    int out_rows_spec = 4;
    /// Negate the imaginary part of every unpacked data sample: the CHORD F-engine's output is
    /// conjugated relative to the gnss44 decode (see GnssChordDequantize.cpp). A flag rather
    /// than a change to gnss44, whose pack and unpack must stay inverses. Read only by
    /// launch_correlate_nm (cudaGnssChordDespread.cu); the despread and peel kernels here
    /// ignore it.
    int conj_data = 0;
    /// Bench option, read only by the waveform kernels in cudaGnssChordDespread.cu (set by
    /// scripts/gnss/wavebench.cpp): an [n_job][n_hops] permutation of the hop index, lane slot
    /// s processing hop hop_perm[b*n_hops + s]. Sorting hops by fractional code phase keeps a
    /// warp's taps within a few Phi entries rather than spread over inv_cps of them, because
    /// each lane's offset base = frac(C_P)*inv_cps is fixed for the whole chip walk and the
    /// d*kf term is common to every lane. null = identity. `wave` is unchanged; only the
    /// energy reduction's lane grouping moves.
    const int* hop_perm = nullptr;
    /// Row stride, in hops, of the [n_chan][data_stride] data array: n_hops for a packed
    /// per-record buffer, or more when the window is read in place from a longer buffer, in
    /// which case the window must be contiguous within each channel row.
    int data_stride;
    /// Use the shared, Doppler-free Phi/Psi tables (DespreadJob::psiA). launch_waveform takes a
    /// separate kernel instantiation, so the per-PRN path keeps its register budget (registers
    /// cap the block width, which sets the DRAM traffic); the despread kernel here branches on
    /// it at runtime; the peel kernel ignores it.
    bool shared = false;
    /// fp16 Phi tables: job.phiA/phiB point at __half2 storage and @ref launch_waveform takes
    /// the __half2 gather, halving the resident tables (storage error ~3.3e-4 relative,
    /// scripts/gnss/phibits). Only the waveform kernels honour it: launch_despread and
    /// launch_peel read Phi as float2, so GnssCudaDespread refuses those paths while fp16
    /// tables are armed. Mutually exclusive with `shared`, whose gather is fp32-only.
    int phi_half = 0;
};

/// float2 wrappers over gnss44::unpack and gnss44::pack, which define the CHORD 4+4b format
/// (documented in gnss44.hpp). Use these rather than re-implementing the codec.
__host__ __device__ inline float2 unpack_44(unsigned char b, float scale) {
    float re, im;
    gnss44::unpack(b, scale, &re, &im);
    return make_float2(re, im);
}
__host__ __device__ inline unsigned char pack_44(float2 v, float inv_scale, int* railed) {
    return gnss44::pack(v.x, v.y, inv_scale, railed);
}

/**
 * Launch the fused despread. One job in, four output rows out: job b writes rows
 * out_rows_spec*b + 0..3 = Early, Prompt, Late, P_HEAD. Jobs and output rows are counted
 * separately (gnss_gpu::max_specs vs max_jobs).
 * @param data   [n_chan][data_stride] channelized voltage at the window start (device)
 * @param code   shared int8 code table (all PRNs concatenated; jobs carry offsets; device)
 * @param jobs   [n_spec] per-PRN quad parameters incl. per-job Phi table pointers (device)
 * @param corr   out [out_rows_spec*n_spec][n_chan] complex correlations (device)
 * @param energy out [out_rows_spec*n_spec][n_chan] replica energies (device)
 * The caller sums across channels.
 */
/// @param xcorr optional out [4*n_spec][n_chan] reference cross-correlations for the analytic
///        peel add-back: rows 4b+0/1 = <R_P, R_E> / <R_P, R_L> over the whole window, rows
///        4b+2/3 = the same restricted to hops [0, m_head). nullptr skips them; that
///        instantiation has no cross-term accumulators.
///        The head-restricted pair keeps the add-back exact in a window where the overlay or nav
///        symbol flips at the code-period boundary and the peel subtracted two different gains:
///        V[k] = V'[k] + a_head*<R_P 1_head, R_k> + a_tail*(whole - head).
///        The add-back needs the code-triangle value as the PFB renders it, not the textbook 0.5
///        at half a chip: substituting 0.5 leaves an E/L asymmetry in the residual that the DLL
///        reads as a code error. The prompt and E/L replicas are in registers at the same hop,
///        so each term is one MAC. <R_P, R_P> and <R_PH, R_P> are energy rows 1 and 3 and are
///        not repeated here.
cudaError_t launch_despread(const float2* data, const int8_t* code, const DespreadJob* jobs,
                            int n_spec, int n_chan, const DespreadParams& p, double2* corr,
                            double* energy, cudaStream_t stream, double2* xcorr = nullptr);

/// As launch_despread, reading 4+4b bytes, one per (channel, hop), unpacked with @c chan_scale
/// [n_chan] (lsb to volts). Same kernel and numerics; only the voltage load differs.
cudaError_t launch_despread_q(const unsigned char* data, const float* chan_scale,
                              const int8_t* code, const DespreadJob* jobs, int n_spec, int n_chan,
                              const DespreadParams& p, double2* corr, double* energy,
                              cudaStream_t stream, double2* xcorr = nullptr);

/**
 * Voltage peel: resid[ci][m] = voltage(ci, m) - sum_jobs a_{job,seg,ci} * R_job(ci, m).
 *
 * One load of the voltage, every job's prompt replica removed, one store of the residual. It
 * uses the despread's chip gather, so with per-PRN tables and no carrier trim the subtracted
 * waveform is bit-identical to the despread's replica (see PeelJob::wc and PeelJob::psiA).
 *
 * The subtractions are simultaneous, not successive: feed-forward gains do not depend on the
 * data, so they commute, and one thread handles every job with no atomics and no ordering.
 *
 * @param data   the voltage: [n_chan][data_stride] float2, or 4+4b bytes with @c chan_scale.
 *               Pointer is at the window start (as launch_despread expects).
 * @param jobs   [n_job] per-PRN peel parameters (device)
 * @param resid  out [n_chan][n_hops] float2, packed (stride n_hops), so the despread then reads
 *               it with data_stride = n_hops. Float2, not 4+4b: a 30 dB correction is far below
 *               one lsb of the CHORD ring, so re-quantizing the residual would erase the peel.
 * Channels with no job covering them are copied through unchanged.
 */
cudaError_t launch_peel(const float2* data, const int8_t* code, const PeelJob* jobs, int n_job,
                        int n_chan, const DespreadParams& p, float2* resid, cudaStream_t stream);

/// As launch_peel, reading 4+4b bytes (dequantized with @c chan_scale on load).
cudaError_t launch_peel_q(const unsigned char* data, const float* chan_scale, const int8_t* code,
                          const PeelJob* jobs, int n_job, int n_chan, const DespreadParams& p,
                          float2* resid, cudaStream_t stream);

} // namespace gnss_cuda

#endif
