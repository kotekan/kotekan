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
 * @param wave    out [3*n_job][n_chan][n_hops] float2, replica samples (device)
 * @param energy  out [4*n_job][n_chan] double, rows E/P/L/P_HEAD. Element-independent, so this
 *                is the whole of what the record header needs; P_HEAD's energy is the prompt's
 *                own, accumulated over the head hops only.
 */
cudaError_t launch_waveform(const int8_t* code, const DespreadJob* jobs, int n_job, int n_chan,
                            const DespreadParams& p, float2* wave, double* energy,
                            cudaStream_t stream);

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

} // namespace gnss_cuda

#endif // CUDA_GNSS_CHORD_DESPREAD_HPP
