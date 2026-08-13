#ifndef CUDA_GNSS_DESPREAD_KERNEL_HPP
#define CUDA_GNSS_DESPREAD_KERNEL_HPP

#include "../stages/gnss/gnss44.hpp" // the ONE 4+4b codec (CUDA-free; shared with the CPU stage)

#include <cuda_runtime.h>
#include <stdint.h>

/**
 * GPU GNSS batched despread: fused hop-rate replica generation + complex dot product.
 *
 * Port of gnss::ChannelizedReplicaBank::hoprate_stream (the closed-form per-hop channelized
 * replica from precomputed cumulative PFB filter tables) fused with gnss::channelized_despread
 * (data x conj(replica) MAC + replica energy). One launch despreads one record window for a
 * whole batch of (PRN x {Early,Prompt,Late}) correlator trials over the covering channels --
 * the CHORD-scale bulk compute (dish x PRN x subband), subband-local by construction.
 *
 * The Phi tables are Doppler-bucketed: built on the CPU (ChannelizedReplicaBank::hoprate_filter)
 * only when a sat's Doppler moves > refresh_hz, then uploaded; the kernel applies the EXACT
 * current code phase / carrier per job. Validated against the CPU path by
 * tests/cuda_gnss_despread_test (<=1e-5 relative).
 */
namespace gnss_cuda {

/// One PRN's whole correlator quad: everything that varies within a batch. Carries its own
/// (device) Phi table pointers + filter span so ONE launch can mix PRNs from different
/// Doppler buckets -- the G1c cross-PRN batch (one launch per record, all sats).
///
/// ONE JOB EMITS FOUR OUTPUT ROWS (E, P, L, P_HEAD) -- see launch_despread. The Early and Late
/// trials are the prompt displaced by +-ds, and every correlator in the quad shares one carrier
/// (same wc) and one data sample per hop, so the kernel generates the carrier and loads the data
/// ONCE per (PRN, channel, hop) instead of four times. P_HEAD costs nothing beyond an extra
/// accumulator: it is the prompt's own MAC, gated on the hop (the despread MAC measured at 0% of
/// the kernel -- 2026-07-16 ablation).
struct DespreadJob {
    double cp0;          ///< PROMPT code phase (COMBINED-stream chips) at absolute sample 0 ref
    double ds;           ///< Early/Late displacement from the prompt (combined-stream chips):
                         ///< row 0 = cp0-ds, row 1 = cp0, row 2 = cp0+ds
    double cps;          ///< chips per sample incl. code Doppler: eff_chip_rate/fs*(1+sign*f/f_c)
    double inv_cps;      ///< 1/cps (tap-boundary indices via multiply, not the costly FP64 divide)
    double wc;           ///< carrier angular rate: 2*pi*(f_offset + doppler)/fs
    /// CARRIER PHASE AT THE WINDOW'S REFERENCE SAMPLE (p.n0), radians in [0, 2*pi), computed on
    /// the host in LONG DOUBLE. Task #52.
    ///
    /// ⚠️ THIS EXISTS TO KEEP wc AWAY FROM THE ABSOLUTE SAMPLE INDEX. The replica phase is
    /// 2*pi*f*t and t is F-engine uptime: n_abs reaches 1.9e15 samples in a week, so ANY
    /// rounding of the ~1.18 GHz carrier -- the ulp of the (f_offset + doppler) sum is 2.4e-7 Hz
    /// -- becomes tenths of a radian of phase, and it RE-ROLLS every record because the Doppler
    /// is re-propagated per record. Measured offline (scripts/gnss/e2e, 24 realizations per
    /// point): the per-record phase residual is flat at 0.012 rad below ~0.2 days of uptime and
    /// then grows EXACTLY LINEARLY with absolute time -- 0.023 / 0.066 / 0.217 rad at 0.6 / 2 /
    /// 6.8 days. On sky it reads 0.745.
    ///
    /// With this field the kernel evaluates ang0 + wc*(n - n0), where (n - n0) never exceeds one
    /// record (3.4e7 samples). The lever drops by eight orders of magnitude and the same
    /// rounding contributes 1.5e-8 rad. It is the carrier-domain twin of #45 step 6, which gave
    /// the CODE phase a reference so it too stopped being reconstructed from f*t_abs
    /// ([[chord-cp-currency]]: "a phase is not an argument").
    ///
    /// ⚠️ n0 IS THE HOP'S LAST SAMPLE (window_start + fft_len - 1), matching par.n0 and
    /// m_head_for. The first/last-sample convention has produced a 52-chip error before; if this
    /// disagrees with par.n0 the whole record rotates by a constant.
    double ang0;
    int code_offset;     ///< this PRN's offset into the shared code table
    int code_len;        ///< combined-stream code length (chips)
    uint64_t chan_mask;  ///< bit ci set = channel ci is in this PRN's covering set (<=64 chans)
    const float2* phiA;  ///< [n_chan][Lf+1] cumulative filter table, this PRN's Doppler bucket
    const float2* phiB;  ///< (float32: see the kernel's mixed-precision note)
    int n_chips;         ///< chips spanned by this bucket's filter (gather depth per hop)
    int m_head;          ///< row 3 (P_HEAD) = the PROMPT accumulated over hops [0, m_head) only:
                         ///< the record's slice BEFORE the code-period boundary, where the
                         ///< secondary overlay flips sign (gnssRecord.hpp slots 16-18). 0 = emit
                         ///< an all-zero head row (the plain 3-trial contract).
};

/// One PRN's PEEL: reconstruct the PROMPT waveform and subtract it from the voltage.
///
/// A cut-down @ref DespreadJob -- no @c ds (E/L are correlation taps, there is no signal at
/// +-1/2 chip to remove) -- plus the complex gain to subtract. Everything else is identical, and
/// deliberately so: the peel and the despread MUST generate bit-identical replicas, or the
/// analytic add-back (docs/gnss_voltage_peel_live.md) stops being exact.
///
/// THE GAIN IS FEED-FORWARD. It comes from a slow EMA of previous records' tracked gain, carried
/// forward one GPU frame, NOT from a despread of this window. That is what buys the single pass,
/// makes the add-back exact for any gain, and leaves the residual a HONEST depth metric (a gain
/// fitted in-window subtracts its own noise back along R, and the residual reads ~0 = infinite
/// depth -- the degeneracy that makes GnssVoltagePeel's own peel_db unusable).
///
/// TWO GAINS, NOT ONE. The window straddles a code-period boundary at hop @c m_head, and every
/// overlay/nav sign flip our signals have lives exactly there. One gain per window models a
/// sign-flipping signal with a constant and removes almost nothing (2026-07-12: B1C peeled 0.9 dB
/// that way). @c a_head applies to hops [0, m_head), @c a_tail to [m_head, n_hops) -- the same
/// boundary the despread's P_HEAD row uses, computed ONCE on the host and shared.
struct PeelJob {
    double cp0;         ///< PROMPT code phase (COMBINED-stream chips) at absolute sample 0 ref
    double cps;         ///< chips per sample incl. code Doppler (== DespreadJob::cps)
    double inv_cps;     ///< 1/cps
    double wc;          ///< carrier angular rate 2*pi*(f_offset + doppler)/fs
    /// Carrier phase at p.n0 -- see DespreadJob::ang0. MUST be built by the SAME expression:
    /// the peel and the despread have to generate bit-identical replicas or the analytic
    /// add-back stops being exact.
    double ang0;
    int code_offset;    ///< offset into the shared code table
    int code_len;       ///< combined-stream code length (chips)
    uint64_t chan_mask; ///< bit ci set = channel ci is in this PRN's covering set
    const float2* phiA; ///< cumulative filter tables, this PRN's Doppler bucket (as DespreadJob)
    const float2* phiB;
    int n_chips;        ///< chips spanned by this bucket's filter
    int m_head;         ///< code-period boundary hop: a_head below it, a_tail at and above it.
                        ///< 0 = no boundary in this window -> a_tail applies throughout.
    const float2* a_head; ///< [n_chan] gain to subtract, hops [0, m_head)
    const float2* a_tail; ///< [n_chan] gain to subtract, hops [m_head, n_hops)
};

/// Batch-shared geometry.
struct DespreadParams {
    long long n0; ///< absolute sample index of the window's first hop reference (+fft_len-1)
    int fft_len;  ///< samples per hop
    int n_hops;   ///< hops per record (<=256)
    int Lf;       ///< filter length fft_len*num_taps (Phi tables have Lf+1 entries)
    int out_rows_spec = 4; ///< OUTPUT rows per spec in corr/energy. 4 normally; 6 when the frame
                           ///< also carries the peel residual rows, so the despread writes its
                           ///< E/P/L/P_HEAD into the same stride the add-back and the assembler
                           ///< use. (xcorr keeps its own stride of 4 -- it is scratch, not frame.)
    int conj_data = 0; ///< negate the imag of every unpacked data sample. The CHORD F-engine's
                       ///< output is CONJUGATED relative to the gnss44 decode (measured on sky
                       ///< 2026-07-30 -- see GnssChordDequantize.cpp for the evidence). A flag
                       ///< here rather than in gnss44: the airspy chain uses gnss44 as a matched
                       ///< encoder/decoder PAIR, which must stay self-consistent.
    /// BENCH: [n_job][n_hops] permutation of the hop index. Thread lane m processes
    /// hop_perm[b*n_hops + m] instead of m. Sorting hops by their FRACTIONAL code phase puts a
    /// warp's 32 lanes inside ~5 Phi entries instead of ~313, because the per-lane offset
    /// base = frac(C_P)*inv_cps is FIXED for the whole chip loop while the d*kf term is common
    /// to every lane -- so the lanes' relative order in the window never changes, it only slides.
    /// null = identity. `wave` is unaffected (each hop's sample depends only on that hop); only
    /// the ENERGY reduction's lane grouping moves.
    const int* hop_perm = nullptr;
    int data_stride; ///< row stride (hops) of the [n_chan][*] data array: n_hops for a packed
                     ///< per-record staging buffer, or the ring length when a window is read in
                     ///< place from the device ring (phase F: ring_hops is a multiple of n_hops,
                     ///< so a record window is always CONTIGUOUS within a channel row)
};

/**
 * CHORD 4+4b post-PFB voltage format: ONE BYTE per complex sample, REAL in the high nibble, IMAG
 * in the low, each a 4-bit OFFSET-ENCODED integer (stored value - 8 -> -8..+7). Zero = 0x88.
 *
 * WHERE THIS CONTRACT COMES FROM. In CHORD the PFB and the 4-bit quantizer are NOT in kotekan at
 * all -- they live upstream in the RFSoC F-engine, and kotekan only ever CONSUMES 4+4b voltages
 * off the wire. So there is no encoder in this tree to copy; the authoritative spec for what we
 * must PRODUCE is the consumer, kotekan's own correlator reference model
 * (lib/testing/gpuSimulate.cpp: `(b & 0x0f) - 8` imag, `((b & 0xf0) >> 4) - 8` real) -- the decode
 * the CHIME/CHORD X-engine is validated against. Match it byte-for-byte and our fftwEngine stands
 * in for an RFSoC: the subband ring below becomes something a real F-engine could feed directly,
 * and our despread becomes an X-engine that could eat one. (Do NOT take
 * lib/cuda/cudaQuantizeKernel4.cu as the model: that is the FRB *beam* quantizer -- 8
 * TWO'S-COMPLEMENT reals per uint32 with a per-chunk scale+offset, a different format for a
 * different product.)
 *
 * The ENCODER is therefore ours to write, and it has to do what the RFSoC does: the bandpass is
 * not flat, so each channel gets its own scale, chosen to put that bin's NOISE rms at ~2-3 lsb
 * (measured at startup -- see cudaGnssTrack's bandpass measurement). That per-bin scaling is what
 * makes 4 bits enough. GNSS lives far below the noise, so the noise dithers the buried signal, and
 * the quantization error -- uncorrelated with a pseudorandom replica -- averages down like thermal
 * noise instead of biasing the correlation. Measured on real sky: < 0.2 dB C/N0 loss at 2 lsb,
 * < 0.14 at 3 lsb (scratchpad/quant_test.py, GPS L5 / Gal E5a / BeiDou B2a). Prefer 3 lsb for RFI
 * headroom; watch per-channel railfrac.
 */
/// float2 wrappers over the ONE codec (../stages/gnss/gnss44.hpp -- CUDA-free so the CPU quantize
/// stage, which must build in the non-CUDA tree, shares it). Do not re-implement either side here.
__host__ __device__ inline float2 unpack_44(unsigned char b, float scale) {
    float re, im;
    gnss44::unpack(b, scale, &re, &im);
    return make_float2(re, im);
}
__host__ __device__ inline unsigned char pack_44(float2 v, float inv_scale, int* railed) {
    return gnss44::pack(v.x, v.y, inv_scale, railed);
}

/**
 * Launch the fused despread. ONE JOB IN, FOUR OUTPUT ROWS OUT: job b writes rows 4b+0..4b+3 =
 * Early, Prompt, Late, P_HEAD. (Jobs and output rows are counted separately upstream -- see
 * gnss_gpu::max_specs vs max_jobs -- because they no longer march together.)
 * @param data   [n_chan][n_hops] channelized voltage (device)
 * @param code   shared int8 code table (all PRNs concatenated; jobs carry offsets; device)
 * @param jobs   [n_spec] per-PRN quad parameters incl. per-job Phi table pointers (device)
 * @param corr   out [4*n_spec][n_chan] complex correlations (device)
 * @param energy out [4*n_spec][n_chan] replica energies (device)
 * Cross-channel summation is left to the caller (tiny; host or follow-up kernel).
 */
/// @param xcorr optional out [4*n_spec][n_chan] reference CROSS-correlations for the analytic
///        peel add-back: rows 4b+0/1 = <R_P, R_E> / <R_P, R_L> over the whole window, rows
///        4b+2/3 = the same restricted to hops [0, m_head). nullptr = do not compute -- the extra
///        accumulators then compile away entirely, so the un-peeled chains pay nothing.
///        The head-restricted pair is what keeps the add-back exact in a window where the
///        overlay/nav symbol flips at the code-period boundary and the peel therefore subtracted
///        two DIFFERENT gains:  V[k] = V'[k] + a_head*<R_P 1_head,R_k> + a_tail*(whole-head).
///        WHY THESE AND NOT THE ANALYTIC TRIANGLE: the add-back V_full[k] = V'[k] + a*<R_P, R_k>
///        needs the code-triangle value AS THE PFB RENDERS IT, not the textbook 0.5 at 1/2 chip.
///        Substituting 0.5 leaves an E/L asymmetry in the residual that the DLL reads as a code
///        error whose SIGN tracks the true error -- i.e. the loop pushes the wrong way. The
///        prompt and the E/L replicas are both in registers at the same hop, so this is one MAC.
///        (<R_P,R_P> and <R_PH,R_P> are just energy rows 1 and 3; they are not duplicated here.)
cudaError_t launch_despread(const float2* data, const int8_t* code, const DespreadJob* jobs,
                            int n_spec, int n_chan, const DespreadParams& p, double2* corr,
                            double* energy, cudaStream_t stream, double2* xcorr = nullptr);

/// As launch_despread, but reading a CHORD 4+4b ring: one byte per (channel, hop), unpacked with
/// @c chan_scale [n_chan] (lsb -> volts, the inverse of the ingest's scale). Same kernel, same
/// numerics, only the voltage load differs -- the demotion must not fork the despread.
/// BENCH ONLY: run the despread with part of the work ablated, to attribute the kernel's cost.
/// abl: 0 = production (nothing ablated), 1 = NO_MAC (synthesis + carrier only, one add kept so
/// the gather is not eliminated), 2 = NO_SYN (constant replica; carrier + MAC only).
///   synthesis  ~ T(0) - T(NO_SYN)
///   correlation~ T(0) - T(NO_MAC)
/// The production launchers instantiate the un-ablated path and are unaffected. Results from an
/// ablated run are MEANINGLESS as despread output -- timing only.
cudaError_t launch_despread_abl(const float2* data, const int8_t* code, const DespreadJob* jobs,
                                int n_spec, int n_chan, const DespreadParams& p, double2* corr,
                                double* energy, cudaStream_t stream, int abl);

cudaError_t launch_despread_q(const unsigned char* data, const float* chan_scale,
                              const int8_t* code, const DespreadJob* jobs, int n_spec, int n_chan,
                              const DespreadParams& p, double2* corr, double* energy,
                              cudaStream_t stream, double2* xcorr = nullptr);

/**
 * VOLTAGE PEEL: resid[ci][m] = voltage(ci, m) - sum_jobs a_{job,seg,ci} * R_job(ci, m).
 *
 * The fused peel of docs/gnss_voltage_peel_live.md, and CHORD path B's "subtract a e^{jphi} R as
 * each X_i streams to registers" made literal: one load of the voltage, every seeded PRN's prompt
 * replica removed, one store of the residual. Reuses the despread kernel's chip gather verbatim,
 * so the subtracted waveform is bit-identical to the one the despread correlates against.
 *
 * SIMULTANEOUS, NOT SUCCESSIVE. The CPU GnssVoltagePeel peels PRNs one after another off a
 * running residual, because its gain is fitted in-window and would otherwise see the other sats'
 * cross-talk. A feed-forward gain does not depend on the data at all, so the subtractions are
 * independent and commute -- which is what lets every PRN be handled inside one thread's loop
 * with no atomics and no ordering.
 *
 * @param data   the voltage: [n_chan][data_stride] float2, or 4+4b bytes with @c chan_scale.
 *               Pointer is at the WINDOW start (as launch_despread expects).
 * @param jobs   [n_job] per-PRN peel parameters (device)
 * @param resid  out [n_chan][n_hops] float2, PACKED (stride n_hops) -- the despread then reads it
 *               with data_stride = n_hops. FLOAT2, NEVER 4+4b: a 30 dB correction is far below
 *               one lsb of the CHORD ring, so re-quantizing the residual would erase the peel.
 * Channels with no job covering them are copied through unchanged.
 */
cudaError_t launch_peel(const float2* data, const int8_t* code, const PeelJob* jobs, int n_job,
                        int n_chan, const DespreadParams& p, float2* resid, cudaStream_t stream);

/// As launch_peel, reading the CHORD 4+4b ring (dequantized with @c chan_scale on load).
cudaError_t launch_peel_q(const unsigned char* data, const float* chan_scale, const int8_t* code,
                          const PeelJob* jobs, int n_job, int n_chan, const DespreadParams& p,
                          float2* resid, cudaStream_t stream);

/**
 * THE ANALYTIC ADD-BACK, in place, right after a despread that ran on the peel residual.
 *
 *     V[k] = V'[k] + a_head*<R_P 1_head, R_k> + a_tail*(<R_P, R_k> - <R_P 1_head, R_k>)
 *
 * Rows 0-3 (E, P, L, P_HEAD) arrive holding V' and leave holding V -- the FULL, un-peeled
 * correlation -- so the assembler, combiner, broker, viewer and TEC chain never learn that a peel
 * happened. The residual they would otherwise lose is preserved first, into rows 4 and 5
 * (gnssRecord.hpp slots 20-23), where it becomes the peel-depth observable.
 *
 * Exact for ANY gain, because on this block one side is the known reference: the peel's
 * contribution to <X, R_k> is data-INDEPENDENT. A wrong (even sign-flipped) feed-forward gain
 * therefore costs residual depth and nothing else -- the tracking loop is untouched.
 *
 * Runs on the host side of the pipeline's cadence but on the GPU: it is O(n_spec x n_chan) with
 * no gathers, far too small to be worth a device->host round trip.
 *
 * @param corr    in/out [rows_spec*n_spec][n_chan], rows_spec = gnss_gpu::ROWS_PEEL
 * @param energy  in     same layout: row 1 = <R_P,R_P>, row 3 = <R_P,R_P>|head
 * @param xcorr   in     [4*n_spec][n_chan] from launch_despread: <R_P,R_E>, <R_P,R_L>, + head
 * @param gains   in     [2*n_spec][n_chan] float2, exactly what launch_peel subtracted
 */
cudaError_t launch_peel_addback(double2* corr, const double* energy, const double2* xcorr,
                                const float2* gains, int n_spec, int n_chan, int rows_spec,
                                cudaStream_t stream);

/// launch_chan_ingest for the 4+4b ring: transpose one hop-major byte frame ([hop][chan], as
/// GnssQuantize44 emits it) into the channel-major ring. A PURE byte transpose -- quantization
/// happens UPSTREAM on the CPU (GnssQuantize44, the RFSoC's job), where CHORD's boundary actually
/// is; doing it there also shrinks the H2D copy 8x. The scales the bytes were encoded with arrive
/// via GnssChanMetadata::chan_scale and feed launch_despread_q.
cudaError_t launch_chan_ingest_q(const unsigned char* frame, unsigned char* ring, int n_hops_f,
                                 int n_chan, long long ring_hops, long long write_hop,
                                 cudaStream_t stream);

/// launch_ring_zero for the 4+4b ring: fills the OFFSET-ENCODED zero (0x88), not 0x00 -- 0x00
/// decodes to (-8,-8), i.e. a full-scale DC spike where a valve drop should read silence.
cudaError_t launch_ring_zero_q(unsigned char* ring, int n_chan, long long ring_hops,
                               long long write_hop, long long count, cudaStream_t stream);

/**
 * Transpose-ingest one hop-major channelized frame into the channel-major device ring
 * (phase F ingest command). frame is [n_hops_f][n_chan] (as the F-engine/bufferRecv delivers),
 * ring is [n_chan][ring_hops]; element (m, c) lands at ring[c][(write_hop + m) % ring_hops].
 */
cudaError_t launch_chan_ingest(const float2* frame, float2* ring, int n_hops_f, int n_chan,
                               long long ring_hops, long long write_hop, cudaStream_t stream);

/// Zero-fill @c count hops of every channel row starting at ring hop @c write_hop (mod
/// ring_hops) -- valve-drop gap fill, so ring position stays identically (absolute hop - hop0)
/// and a window overlapping a gap despreads against clean zeros (SNR loss, never misalignment).
cudaError_t launch_ring_zero(float2* ring, int n_chan, long long ring_hops, long long write_hop,
                             long long count, cudaStream_t stream);

} // namespace gnss_cuda

#endif
