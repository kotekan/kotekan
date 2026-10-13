/**
 * @file
 * @brief CHORD N-antenna despread: replica synthesis split from the N x M correlation.
 *
 * The fused despread (@ref launch_despread) synthesises each replica sample inside the
 * correlation MAC, which suits a single input. With N antennas the same replica is correlated
 * against every one, so it is generated once (@ref launch_waveform) and then either correlated
 * against the antennas here (@ref launch_correlate_nm) or, on path B, quantized into synthetic
 * stations of the N^2 correlator (@ref launch_pack44, cudaGnssInject, cudaCorrelatorDual).
 *
 * The synthesis calls the same gather, carrier and combination as the fused kernel
 * (cudaGnssReplicaDevice.cuh) and takes the same @ref gnss_cuda::DespreadJob.
 * cudaGnssDespreadTest requires the split and fused paths to agree exactly at N = 1; a
 * divergence would otherwise show only as a correlation-amplitude error.
 *
 * Replica energy is element-independent and goes in the per-PRN record header; correlations
 * are per antenna and go in the element blocks (gnssRecord.hpp). Path B's assembler
 * (GnssN2RecordAssemble) fills the same layout from the N^2 tiles.
 */

#ifndef CUDA_GNSS_CHORD_DESPREAD_HPP
#define CUDA_GNSS_CHORD_DESPREAD_HPP

#include "cudaGnssDespreadKernel.hpp" // for DespreadJob, DespreadParams

#include <cuda_runtime.h>

namespace gnss_cuda {

/// Block width requested by @ref launch_waveform; it sets the synthesis kernel's DRAM traffic.
/// Each block walks its (PRN, channel) Phi slice once per hop pass, n_hops/W passes for a block
/// of W threads. At CHORD geometry the walked part of a slice is several times L1, so every walk
/// streams it from DRAM again. With per-PRN tables (Doppler-bucketed in GnssCudaDespread's
/// PhiCache) no two blocks share a slice. @c wave is bit-identical at every width; the energy
/// rows agree to rounding (lane L sums hops L, L+W, L+2W, ..., then a W-lane tree). The width is
/// capped at n_hops, rounded down to a power of two, and at what the driver reports the
/// instantiation can launch. cudaGnssDespreadTest's default 125 hops launch 64 threads, so the
/// test does not exercise the 512- and 1024-wide launches.
static constexpr int WAVE_THREADS = 1024;

/**
 * @brief Synthesise the E/P/L replicas for a batch of PRNs over the covering channels.
 *
 * Each job writes three replica rows, E, P, L (0, 1, 2), in the fused kernel's accumulator
 * order. P_HEAD is not a fourth row: it is the prompt over hops [0, m_head), so consumers
 * (launch_correlate_nm, launch_pack44) gate the prompt row instead.
 *
 * @param code    batch-shared code table (device)
 * @param jobs    [n_job] per-PRN geometry, the same struct the fused path takes (device)
 * @param n_job   PRNs in this batch
 * @param n_chan  covering channels
 * @param p       batch-shared geometry (n0, fft_len, n_hops, Lf); @c phi_half selects the
 *                table format
 * @param wave    out [3*n_job][n_chan][n_hops] float2 replica samples (device). Hop @c mh is
 *                stored at index @c mh, in time order: path B's pack maps the wave index to the
 *                N^2 correlator's time sample, so the hop axis must not be permuted
 *                (gnssRecord.hpp).
 * @param rot     nullptr, or [n_job][n_chan] float radians: the E row is then the prompt
 *                rotated by +rot and the L row by -rot per channel, from one chip walk instead
 *                of three, and their energy rows equal the prompt's. A code offset of ds chips
 *                is a phase rotation of the prompt on the channel centred at f_c,
 *                exp(+i 2 pi (f_c - f_car) ds / chip_rate) for the prompt - ds trial, with
 *                f_car the job's carrier (DespreadJob::wc), up to the offset's phase ramp
 *                across one channel (0.06 rad at CHORD's 195 kHz channels and 0.5 chips of
 *                10.23 Mcps). GnssCudaDespread::set_epl_from_prompt computes it;
 *                cudaGnssDespreadTest pins the sign against the fused kernel's E/L.
 * @param energy  out [4*n_job][n_chan] double, rows E/P/L/P_HEAD; P_HEAD is the prompt's energy
 *                over the head hops only. Uncovered channels get zero replica and zero energy.
 */
cudaError_t launch_waveform(const int8_t* code, const DespreadJob* jobs, int n_job, int n_chan,
                            const DespreadParams& p, const float* rot, float2* wave, double* energy,
                            cudaStream_t stream);

/// Host-side float2 -> __half2 conversion of @c n Phi entries into @c dst (n * 4 bytes); void*
/// keeps __half2 out of g++-compiled callers. GnssCudaDespread falls back to it when built
/// without F16C/AVX.
void phi_to_half(const float2* src, void* dst, size_t n);

/**
 * @brief Correlate N antenna voltages against the replicas from @ref launch_waveform.
 *
 * Reads the CHORD voltage in its native layout, [hop][frame_chan][element] with element
 * fastest, so there is no transpose: consecutive threads take consecutive elements (coalesced),
 * and the replica sample for a (row, channel, hop) is the same for all of them (a broadcast).
 * As in the fused kernel, one block produces a PRN's rows (here E/P/L/P_HEAD) from one data
 * load.
 *
 * @param data        [n_hops][frame_chan_stride][elem_stride] 4+4b bytes (device), real in the
 *                    high nibble, imag in the low (gnss44.hpp)
 * @param chan_scale  [n_chan] lsb -> volts per covering channel
 * @param chan_ids    [n_chan] index of each covering channel within the frame's channel axis
 * @param wave        [3*n_job][n_chan][n_hops] float2 from @ref launch_waveform
 * @param jobs        [n_job]; only @c m_head is read (the P_HEAD gate)
 * @param n_elem      antennas correlated (threads span this)
 * @param elem_stride element axis stride of the frame (>= n_elem)
 * @param frame_chan_stride channel axis stride of the frame (num_local_freq)
 * @param p           n_hops, and @c conj_data, which negates the data's imag on load
 * @param corr        out [4*n_job][n_chan][n_elem] double2, sum of data * conj(replica), rows
 *                    E/P/L/P_HEAD
 */
cudaError_t launch_correlate_nm(const unsigned char* data, const float* chan_scale,
                                const int* chan_ids, const float2* wave, const DespreadJob* jobs,
                                int n_job, int n_chan, int n_elem, int elem_stride,
                                int frame_chan_stride, const DespreadParams& p, double2* corr,
                                cudaStream_t stream);

/**
 * @brief Path B: quantize @ref launch_waveform's replicas into synthetic 4+4b N^2 stations.
 *
 * Writes one record's rows of the synthetic input array (cudaCorrelatorDual's @c gnss_synth),
 * [hop][channel][lane] with lane = 4*prn_slot + trial (E/P/L/P_HEAD; the head lane is the
 * prompt, zeroed from hop m_head on). Inactive slots, uncovered channels and the gated head
 * region get 0x88 (0+0j). Values clamp to [-7,7]: -8 corrupts the N^2 (n2k's negate_4bit).
 * The scale is 7/(3*rms) per (lane, channel), rms = sqrt(energy row / n_hops), except that the
 * P_HEAD lane divides by the job's m_head. The path-B consumer (GnssN2RecordAssemble)
 * recomputes the scale from the energy rows over hops_per_record and divides it out; for the
 * P_HEAD lane its scale differs from this one.
 *
 * The replica's imag is negated before quantizing, the path-B counterpart of
 * @ref DespreadParams::conj_data: the N^2 kernel has no conjugate flag and the antenna input is
 * shared, so CHORD's F-engine conjugation is absorbed on the replica side. It cannot be fixed
 * downstream: the correlator forms sum_t R conj(D) and the tracker sum_t conj(D) conj(R), which
 * are not conjugates of each other.
 *
 * @param wave       [3*n_job][n_chan][n_hops] float2, from @ref launch_waveform
 * @param energy     [4*n_job][n_chan] double, rows E/P/L/P_HEAD, from @ref launch_waveform.
 *                   cudaGnssInject passes the frame's first record for every record, so the
 *                   E/P/L scales are constant over a frame.
 * @param jobs       [n_job] device jobs (m_head, chan_mask)
 * @param slot2spec  [n_slot] device: job index of each PRN slot this record, -1 = inactive
 * @param chan_map   [n_chan] device: position of each covering channel on the synth array's
 *                   channel axis
 * @param frame_chan_stride length of the synth array's channel axis
 * @param n_lanes    lanes this chain writes (4 per slot); all are written, inactive slots as 0x88
 * @param lane_base  first lane of this chain on the synth array's lane axis (a multiple of 16,
 *                   one tile row), so several chains can share one array
 * @param lane_pitch length of the synth array's lane axis
 * @param synth      base of this record's rows: [n_hops][frame_chan_stride][lane_pitch] bytes
 */
cudaError_t launch_pack44(const float2* wave, const double* energy, const DespreadJob* jobs,
                          const int* slot2spec, int n_slot, int n_chan, int n_hops,
                          const int* chan_map, int frame_chan_stride, int n_lanes, int lane_base,
                          int lane_pitch, unsigned char* synth, cudaStream_t stream);

} // namespace gnss_cuda

#endif // CUDA_GNSS_CHORD_DESPREAD_HPP
