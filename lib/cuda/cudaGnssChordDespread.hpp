/**
 * @file
 * @brief CHORD N-antenna despread: replica generation SPLIT from N x M correlation.
 *
 * The airspy kernel (@ref launch_despread) fuses replica synthesis with the correlation MAC.
 * That is right at N=1, where each replica sample is consumed exactly once and materialising
 * it would be pure overhead. It is WRONG at N>1: the same replica is correlated against every
 * antenna, so fusing means either regenerating it N times or holding it live across the
 * antenna loop. Generate once, reuse N times is strictly better the moment N>1, and the
 * advantage grows with N.
 *
 * So the split here is not scaffolding to be thrown away. The waveform generator is the piece
 * that SURVIVES: the end state (docs/gnss_chord_framework.md "path B") injects replicas as
 * synthetic inputs into CHORD's N^2 kernel, where the correlation rides the optimised
 * (N+M)^2 kernel and the N voltages are loaded once. Only the CONSUMER changes -- from
 * @ref launch_correlate_nm to synthetic lanes. This standalone correlator is the framework
 * doc's "path A", which it calls simpler to implement and debug, and at 32 antennas the extra
 * voltage load it costs is irrelevant.
 *
 * BIT-IDENTICAL REPLICAS. The generator calls the same gather, carrier and combination as the
 * fused kernel (cudaGnssReplicaDevice.cuh), takes the same @ref gnss_cuda::DespreadJob, and is
 * gated by a test that runs both paths on identical inputs and requires exact agreement. If
 * they ever diverge it shows up only as a correlation-amplitude discrepancy, with no other
 * symptom -- so it is checked rather than assumed.
 *
 * WHAT LIVES WHERE, and why it matches the record schema. Replica ENERGY is element-
 * independent (one replica against every antenna), so the generator computes it and it lands
 * in the per-PRN record header. The CORRELATIONS are per-antenna and land in the element
 * blocks (gnssRecord.hpp). That is the same division path B gets for free -- N x M is the
 * element blocks, M x M is the header energies -- which is why the schema does not change when
 * the consumer is swapped.
 */

#ifndef CUDA_GNSS_CHORD_DESPREAD_HPP
#define CUDA_GNSS_CHORD_DESPREAD_HPP

#include "cudaGnssDespreadKernel.hpp" // for DespreadJob, DespreadParams

#include <cuda_runtime.h>

namespace gnss_cuda {

/**
 * @brief Synthesise the E/P/L replicas for a batch of PRNs over the covering channels.
 *
 * One job emits THREE replica rows -- Early, Prompt, Late -- in the fused kernel's own
 * accumulator order (0 = E, 1 = P, 2 = L), so a row index means the same thing in both paths.
 * P_HEAD is deliberately NOT a fourth row: it is the prompt restricted to hops [0, m_head),
 * i.e. the same samples, so materialising it would duplicate the prompt. The correlator gates
 * the prompt's accumulation instead, exactly as the fused kernel does.
 *
 * @param code    batch-shared code table (device)
 * @param jobs    [n_job] per-PRN geometry -- the SAME struct the fused path takes
 * @param n_job   PRNs in this batch
 * @param n_chan  covering channels
 * @param p       batch-shared geometry (n0, fft_len, n_hops, Lf)
 * @param wave    out [3*n_job][n_chan][n_hops] float2, replica samples (device). HOP-ORDER
 *                CONTRACT: hop `mh` lives at index `mh` (time order) -- an invariant, not a
 *                convenience. Path B's pack kernel maps wave index == N^2 correlator time
 *                sample, so sorted-hop storage (10.6b) is disallowed; see gnssRecord.hpp.
 * @param energy  out [4*n_job][n_chan] double, rows E/P/L/P_HEAD. Element-independent, so this
 *                is the whole of what the record header needs; P_HEAD's energy is the prompt's
 *                own, accumulated over the head hops only.
 */
/// Block width for @ref launch_waveform, and THE tuning knob for the synthesis kernel.
///
/// It is not an occupancy setting -- it is a MEMORY setting. Each block owns one (PRN, channel)
/// Phi slice, and NO TWO BLOCKS SHARE ONE: the tables are Doppler-bucketed per PRN (see
/// GnssCudaDespread::Impl::PhiCache), because the carrier rate sits inside the table's exponent.
/// The block walks its slice once per HOP PASS and once per E/P/L TRIAL, and between two visits
/// to a given chip it has streamed the whole 689 KB slice through a 128 KB L1, so nothing
/// survives. A block of W threads makes n_hops/W passes, hence 3*n_hops/W re-walks, so the
/// kernel's DRAM traffic is INVERSELY PROPORTIONAL TO W. At CHORD's 2048 hops the old 256 gave
/// 24 re-walks of a 689 KB slice = 1.24 GB per launch against 52 MB of distinct data.
///
/// Measured on cx19 (A40), 11 PRNs x 7 channels x 140 chips, scripts/gnss/wavebench.cpp:
///
///     256 threads  2.519 ms  1.00x       512 threads  1.234 ms  2.03x
///                                       1024 threads  0.684 ms  3.67x
///
/// `wave` is BIT-IDENTICAL at every width -- each hop's replica sample depends only on that hop,
/// so W moves which thread computes it and nothing else. Raising it costs only the 8 KB sh_e
/// array, and the block is capped by n_hops in any case: at the unit test's 125 hops the launch
/// is byte-for-byte the old one, which is why the bench and not the test is the gate here.
///
/// This is a REQUEST. The real ceiling is registers (see the kernel's __launch_bounds__ note) and
/// @ref launch_waveform_tuned clamps to what the driver reports.
static constexpr int WAVE_THREADS = 1024;

cudaError_t launch_waveform(const int8_t* code, const DespreadJob* jobs, int n_job, int n_chan,
                            const DespreadParams& p, float2* wave, double* energy,
                            cudaStream_t stream);

/// Walk the chip loop ONCE for all three E/P/L trials instead of three times -- the other half of
/// the re-walk count @ref WAVE_THREADS attacks. Bit-identical, but it costs registers, and
/// registers are what cap the block width, so the two knobs pull against each other. See
/// gnss_cuda::chip_gather3 and scripts/gnss/wavebench.cpp for which setting wins where.
static constexpr bool WAVE_FUSE3 = true;

/// @ref launch_waveform with the block width and the trial fusion overridden. BENCH ONLY.
///
/// @param threads_hint block width, 0 = @ref WAVE_THREADS. The width changes the ENERGY
///        summation order (lane L sums hops L, L+W, L+2W, ... then a W-lane tree), so two widths
///        agree to rounding, not bit-for-bit. @c wave is unaffected -- each hop's replica sample
///        depends only on that hop.
/// @param fuse3 -1 = @ref WAVE_FUSE3, 0 = three separate gathers, 1 = fused. Bit-identical either
///        way, energy included: each trial's accumulation over chips is untouched, only the
///        interleaving of the three moves.
/// @param phi16 BENCH ONLY. 1 = read Phi as __half2 (the caller must have uploaded half tables
///        and stashed the pointer in job.phiA). Forces the 512-wide fused instantiation.
cudaError_t launch_waveform_tuned(const int8_t* code, const DespreadJob* jobs, int n_job,
                                  int n_chan, const DespreadParams& p, float2* wave, double* energy,
                                  int threads_hint, int fuse3, int phi16, cudaStream_t stream);

/**
 * @brief Correlate N antenna voltages against the M generated references.
 *
 * Reads the CHORD voltage in its NATIVE layout -- [hop][frame_chan][element], element fastest
 * -- so there is no transpose. Consecutive threads take consecutive elements, which is fully
 * coalesced, and the replica sample for a given (row, channel, hop) is uniform across the
 * block, so it broadcasts.
 *
 * ONE PRN'S QUAD PER BLOCK, mirroring the fused kernel: E/P/L/P_HEAD share one data load, so
 * the voltage is read n_job times per record rather than 4*n_job. At CHORD scale even that is
 * the traffic path B removes entirely by folding the despread into the N^2 kernel's single
 * load; here it is deliberate and cheap.
 *
 * @param data        [n_hops][frame_chan_stride][elem_stride] 4+4b bytes (device). HIGH nibble
 *                    is REAL, low is imag -- see cudaGnssDespreadKernel.hpp's codec note.
 * @param chan_scale  [n_chan] lsb -> volts per covering channel
 * @param chan_ids    [n_chan] index of each covering channel within the frame's channel axis
 * @param wave        [3*n_job][n_chan][n_hops] float2 from @ref launch_waveform
 * @param jobs        [n_job] -- only @c m_head is read (the P_HEAD gate); everything else about
 *                    the replica is already baked into @c wave
 * @param n_elem      antennas actually correlated (threads span this)
 * @param elem_stride element axis stride of the frame (>= n_elem; CHORD allocates 128 with far
 *                    fewer live, so these differ)
 * @param frame_chan_stride channel axis stride of the frame (num_local_freq)
 * @param corr        out [4*n_job][n_chan][n_elem] double2, rows E/P/L/P_HEAD
 */
cudaError_t launch_correlate_nm(const unsigned char* data, const float* chan_scale,
                                const int* chan_ids, const float2* wave, const DespreadJob* jobs,
                                int n_job, int n_chan, int n_elem, int elem_stride,
                                int frame_chan_stride, const DespreadParams& p, double2* corr,
                                cudaStream_t stream);

/**
 * @brief Path B: quantize @ref launch_waveform's replicas into synthetic 4+4b N^2 stations.
 *
 * Writes one record's worth of the synthetic input array (cudaCorrelatorDual's
 * @c gnss_synth): lanes at [hop][frame_chan][lane], lane = 4*prn_slot + trial
 * (E/P/L/P_HEAD -- the head lane is the prompt zeroed at and beyond m_head). Slots without a
 * live spec, non-covered channels and the gated head region get 0x88 (0+0j). Values clamp to
 * [-7,7]: -8 silently corrupts the N^2 (n2k negate_4bit). Scale = 7/(3*rms) per (lane, chan)
 * from @c energy; consumers normalize by the M^2 diagonal, so the scale cancels.
 *
 * @param wave       [3*n_job][n_chan][n_hops] float2, from @ref launch_waveform
 * @param energy     [4*n_job][n_chan] double, rows E/P/L/P_HEAD (same launch)
 * @param jobs       [n_job] device jobs (m_head, chan_mask)
 * @param slot2spec  [n_slot] device: job index of each PRN slot this record, -1 = inactive
 * @param chan_map   [n_chan] device: LOCAL frame channel of each covering channel
 * @param synth      base of THIS RECORD's rows: [n_hops][frame_chan_stride][num_synth] bytes
 */
cudaError_t launch_pack44(const float2* wave, const double* energy, const DespreadJob* jobs,
                          const int* slot2spec, int n_slot, int n_chan, int n_hops,
                          const int* chan_map, int frame_chan_stride, int num_synth,
                          unsigned char* synth, cudaStream_t stream);

} // namespace gnss_cuda

#endif // CUDA_GNSS_CHORD_DESPREAD_HPP
