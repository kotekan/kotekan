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
/// One job emits four output rows (E, P, L, P_HEAD); see launch_waveform and launch_despread.
/// Early and Late are the prompt displaced by +-ds, and all three share one carrier (same wc)
/// and one data sample per hop, so the kernels generate the carrier and load the data once per
/// (PRN, channel, hop). P_HEAD is the prompt gated on the hop (see m_head).
struct DespreadJob {
    double ds; ///< Early/Late displacement from the prompt (combined-stream chips):
               ///< row 0 = prompt-ds, row 1 = prompt, row 2 = prompt+ds
    /// Prompt code phase at the window's reference sample p.n0, combined-stream chips, reduced
    /// mod code_len on the host in long double.
    ///
    /// Evaluated in double on the absolute sample, cp0 + n*cps (cp0 the prompt code phase
    /// referred to absolute sample 0) reaches ~6e12 chips at n ~ 1.9e15 (about a week of
    /// F-engine uptime), where a double's ulp is ~1e-3 chips, and
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
    /// [n_chan][Lf+1] cumulative filter tables for this PRN's Doppler bucket: float2, or
    /// __half2 storage when DespreadParams::phi_half is set.
    const float2* phiA;
    const float2* phiB;
    int n_chips; ///< chips spanned by this bucket's filter; the gather walks [d_first, n_chips)
    int m_head;  ///< row 3 (P_HEAD) is the prompt over hops [0, m_head) only: the part of the
                 ///< window before the code-period boundary, where the secondary overlay or nav
                 ///< symbol can flip sign (gnssRecord.hpp REC_PH_*). 0 gives an all-zero row 3.
                 ///< Read by the waveform path (launch_waveform, launch_correlate_nm,
                 ///< launch_pack44); launch_despread ignores it and writes row 3 as zeros.
    /// First chip of the gather walk, for centered truncation; 0 walks the span from chip 0.
    /// With n_chips capped, (d_first, n_chips) selects the central window of the PFB span,
    /// where the prototype peaks. Set from the same tables as n_chips. Last because the jobs
    /// are built with positional aggregate initializers (GnssCudaDespread.cpp,
    /// cudaGnssDespreadTest.cpp), and a field inserted earlier would shift every later
    /// initializer.
    int d_first = 0;
};

/// Batch-shared geometry.
struct DespreadParams {
    long long n0; ///< reference sample of hop 0, window_start + fft_len - 1 (the hop's last
                  ///< sample); the host evaluates cp_ref and ang0 here
    int fft_len;  ///< samples per hop
    int n_hops;   ///< hops per record
    int Lf;       ///< filter length fft_len*num_taps (Phi tables have Lf+1 entries)
    /// Negate the imaginary part of every unpacked data sample: the CHORD F-engine's output is
    /// conjugated relative to the gnss44 decode (see GnssChordDequantize.cpp). A flag rather
    /// than a change to gnss44, whose pack and unpack must stay inverses. Read only by
    /// launch_correlate_nm (cudaGnssChordDespread.cu); the despread kernel here ignores it.
    int conj_data = 0;
    /// Row stride, in hops, of the [n_chan][data_stride] data array: n_hops for a packed
    /// per-record buffer, or more when the window is read in place from a longer buffer, in
    /// which case the window must be contiguous within each channel row.
    int data_stride;
    /// fp16 Phi tables: job.phiA/phiB point at __half2 storage and @ref launch_waveform takes
    /// the __half2 gather, halving the resident tables (storage error ~3.3e-4 relative,
    /// scripts/gnss/phibits). Only the waveform kernels honour it: launch_despread reads Phi
    /// as float2, so GnssCudaDespread refuses that path while fp16 tables are armed.
    int phi_half = 0;
};

/// float2 wrapper over gnss44::unpack, which defines the CHORD 4+4b format (documented in
/// gnss44.hpp). Use it rather than re-implementing the codec.
__host__ __device__ inline float2 unpack_44(unsigned char b, float scale) {
    float re, im;
    gnss44::unpack(b, scale, &re, &im);
    return make_float2(re, im);
}

/**
 * Launch the fused despread. One job in, four output rows out: job b writes rows
 * 4*b + 0..2 = Early, Prompt, Late, and row 4*b + 3 as zeros (the P_HEAD slot, which only the
 * waveform path fills). Jobs and output rows are counted separately (gnss_gpu::max_specs vs
 * max_jobs).
 * @param data   [n_chan][data_stride] channelized voltage at the window start (device)
 * @param code   shared int8 code table (all PRNs concatenated; jobs carry offsets; device)
 * @param jobs   [n_spec] per-PRN quad parameters incl. per-job Phi table pointers (device)
 * @param corr   out [4*n_spec][n_chan] complex correlations (device)
 * @param energy out [4*n_spec][n_chan] replica energies (device)
 * The caller sums across channels.
 */
cudaError_t launch_despread(const float2* data, const int8_t* code, const DespreadJob* jobs,
                            int n_spec, int n_chan, const DespreadParams& p, double2* corr,
                            double* energy, cudaStream_t stream);

} // namespace gnss_cuda

#endif
