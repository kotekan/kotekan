/**
 * @file cudaPilotProxyPacker.hpp
 * @brief PilotProxy detector-input packer kernel (A40 / sm_86).
 *
 * Extracts one coarse frequency channel from the CHORD voltage ring buffer
 * ([T, F, P, D] @c int4x2_swapped_withoffset bytes), converts each byte from
 * offset-binary to the PilotProxy kernel's two's-complement packed complex
 * int4 layout, and materializes the row-major detector-matrix view expected
 * by the vendored F-statistic kernel (external/pilotproxy/f_statistic.h):
 *
 * @code
 * packed_out[(s * windows + w) * K + k]
 *     = voltage[(pos + w*K + k') mod ring, f, p, d] XOR 0x88
 * @endcode
 *
 * with stream <tt>s = p * num_dishes + d</tt>, @c K = @c
 * detector_window_samples, and <tt>k' = k</tt> or <tt>k' = K-1-k</tt>
 * (@c time_reverse_windows). The reversal is the adapter flip assumed by
 * post-spectral-sense weight banks (<tt>exp(-2j*pi*f*k)</tt> templates in
 * the true-sense raw frame). The runtime bundle requests it via
 * @c input_preprocessing, and CHORD bundles set it true despite the upright
 * spectral sense. The XOR flips both nibble sign bits: kotekan's
 * @c int4x2_swapped_withoffset stores value+8 per nibble with the imaginary
 * component in the low nibble and the real component in the high nibble,
 * which is the PilotProxy contract's component order, so the offset removal
 * is the entire conversion (lossless; verified against the pilot-proxy
 * CHIME frame adapter semantics). The 0x88 offset-encoding XOR is the same
 * in-tree idiom n2k uses for these voltages (external/n2k
 * src_lib/internals.cu, @c unpack_e_array).
 *
 * See cudaPilotProxyPacker.cu for the kernel and the CPU reference.
 */

#ifndef CUDA_PILOTPROXY_PACKER_HPP
#define CUDA_PILOTPROXY_PACKER_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief Pack one coarse channel's detector block from the voltage ring
 * buffer (GPU kernel launch).
 *
 * @param packed_out        Device output, int8 detector matrix
 *                          [num_streams * (num_time_samples / K)] x [K],
 *                          num_streams = num_polarizations * num_dishes.
 * @param voltage_ring      Device input ring buffer bytes,
 *                          [ringbuf_size_t][num_frequencies]
 *                          [num_polarizations][num_dishes].
 * @param num_dishes        Dish count D.
 * @param num_polarizations Polarization count P.
 * @param num_frequencies   Local coarse-frequency count F on this node.
 * @param freq_index        Local frequency index to extract, in [0, F).
 * @param num_time_samples  Channelized samples per detector block (multiple
 *                          of detector_window_samples; 8192 deployed).
 * @param ringbuf_size_t    Ring extent in time samples; must be a power of
 *                          two.
 * @param ringbuf_pos_t     Logical start position in time samples (wrapped
 *                          internally with the power-of-two mask).
 * @param detector_window_samples K from the compiled kernel (64 on CHORD;
 *                          128 on CHIME).
 * @param time_reverse_windows Reverse each K-sample window (the adapter
 *                          flip assumed by post-spectral-sense weight
 *                          banks; true for CHORD runtime bundles).
 * @param stream            CUDA stream to launch on.
 */
void launch_pilotproxy_pack(std::int8_t* packed_out, const std::uint8_t* voltage_ring,
                            std::ptrdiff_t num_dishes, std::ptrdiff_t num_polarizations,
                            std::ptrdiff_t num_frequencies, int freq_index,
                            std::ptrdiff_t num_time_samples, std::ptrdiff_t ringbuf_size_t,
                            std::ptrdiff_t ringbuf_pos_t, int detector_window_samples,
                            bool time_reverse_windows, cudaStream_t stream);

/**
 * @brief CPU reference for @c launch_pilotproxy_pack (same arguments, minus
 * the CUDA stream; host pointers).
 *
 * The slow, straightforward CPU oracle used by testPilotProxyDetector. If
 * the GPU kernel disagrees with this reference, the GPU kernel is at
 * fault.
 */
void cpu_pilotproxy_pack(std::int8_t* packed_out, const std::uint8_t* voltage_ring,
                         std::ptrdiff_t num_dishes, std::ptrdiff_t num_polarizations,
                         std::ptrdiff_t num_frequencies, int freq_index,
                         std::ptrdiff_t num_time_samples, std::ptrdiff_t ringbuf_size_t,
                         std::ptrdiff_t ringbuf_pos_t, int detector_window_samples,
                         bool time_reverse_windows);

#endif // CUDA_PILOTPROXY_PACKER_HPP
