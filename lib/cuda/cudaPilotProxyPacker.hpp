// PilotProxy detector-input packer kernel (A40 / sm_86).
//
// Extracts one coarse frequency channel from the CHORD voltage ring buffer
// ([T, F, P, D] int4x2_swapped_withoffset bytes), converts each byte from
// offset-binary to the PilotProxy kernel's two's-complement packed complex
// int4 layout, and materializes the row-major detector-matrix view expected
// by the vendored F-statistic kernel (lib/cuda/pilotproxy/f_statistic.h):
//
//     packed_out[(s * windows + w) * K + k]
//         = voltage[(pos + w*K + k') mod ring, f, p, d] XOR 0x88
//
// with stream s = p * num_dishes + d, K = detector_window_samples, and
// k' = k or k' = K-1-k (time_reverse_windows). The reversal is the adapter
// flip assumed by post-spectral-sense weight banks (exp(-2j*pi*f*k)
// templates in the true-sense raw frame): the runtime bundle requests it
// via input_preprocessing, and CHORD bundles set it true despite the
// upright spectral sense. The XOR flips both nibble sign
// bits: kotekan's int4x2_swapped_withoffset stores value+8 per nibble with
// the imaginary component in the low nibble and the real component in the
// high nibble, which is the PilotProxy contract's component order, so the
// offset removal is the entire conversion (lossless; verified against the
// pilot-proxy CHIME frame adapter semantics).
//
// See cudaPilotProxyPacker.cu for the kernel and the CPU reference.

#ifndef CUDA_PILOTPROXY_PACKER_HPP
#define CUDA_PILOTPROXY_PACKER_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Pack one coarse channel's detector block from the voltage ring buffer.
//
//   packed_out       - device output, int8 detector matrix
//                      [num_streams * (num_time_samples / K)] x [K],
//                      num_streams = num_polarizations * num_dishes
//   voltage_ring     - device input ring buffer bytes,
//                      [ringbuf_size_t][num_frequencies][num_polarizations]
//                      [num_dishes]
//   num_dishes       - dish count D
//   num_polarizations- polarization count P
//   num_frequencies  - local coarse-frequency count F on this node
//   freq_index       - local frequency index to extract, in [0, F)
//   num_time_samples - channelized samples per detector block (multiple of
//                      detector_window_samples; 16384 deployed)
//   ringbuf_size_t   - ring extent in time samples; must be a power of two
//   ringbuf_pos_t    - logical start position in time samples (wrapped
//                      internally with the power-of-two mask)
//   detector_window_samples - K (128 for the locked detector)
//   time_reverse_windows    - reverse each K-sample window (the adapter flip
//                             assumed by post-spectral-sense weight banks;
//                             true for CHORD runtime bundles)
//   stream           - CUDA stream to launch on
void launch_pilotproxy_pack(std::int8_t* packed_out, const std::uint8_t* voltage_ring,
                            std::ptrdiff_t num_dishes, std::ptrdiff_t num_polarizations,
                            std::ptrdiff_t num_frequencies, int freq_index,
                            std::ptrdiff_t num_time_samples, std::ptrdiff_t ringbuf_size_t,
                            std::ptrdiff_t ringbuf_pos_t, int detector_window_samples,
                            bool time_reverse_windows, cudaStream_t stream);

// CPU reference for `launch_pilotproxy_pack` (same arguments, minus the CUDA
// stream; host pointers). The slow, obviously-correct oracle used by
// testPilotProxyDetector; if the GPU kernel disagrees with it, the GPU
// kernel is wrong.
void cpu_pilotproxy_pack(std::int8_t* packed_out, const std::uint8_t* voltage_ring,
                         std::ptrdiff_t num_dishes, std::ptrdiff_t num_polarizations,
                         std::ptrdiff_t num_frequencies, int freq_index,
                         std::ptrdiff_t num_time_samples, std::ptrdiff_t ringbuf_size_t,
                         std::ptrdiff_t ringbuf_pos_t, int detector_window_samples,
                         bool time_reverse_windows);

#endif // CUDA_PILOTPROXY_PACKER_HPP
