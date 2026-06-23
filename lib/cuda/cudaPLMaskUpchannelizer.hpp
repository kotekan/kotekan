// Packet-loss (PL) mask upchannelizer kernel (A40 / sm_86).
//
// Propagates the expanded PL mask onto the upchannelized time grid: each
// upchannelized output sample depends on M*U consecutive input samples (M =
// PFB taps, U = upchannelization factor), so the output sample is valid only
// if *all* M*U contributing input bits are valid -- a logical AND.
//
// See cudaPLMaskUpchannelizer.cu for the full algorithm description.

#ifndef CUDA_PL_MASK_UPCHANNELIZER_HPP
#define CUDA_PL_MASK_UPCHANNELIZER_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// These are free functions (not in a namespace) so the `cudaPLMaskUpchannelizer` cudaCommand class
// can use that name; this mirrors the gpu_/cpu_ pairing in cudaQuantizeKernel8.hpp.

// Upchannelize the expanded PL mask by a factor `U`, over a frequency sub-range.
//
// Array layouts (each `uint64` holds 64 consecutive time bits; rows are frequency-major). The
// input and output are independent power-of-two ring buffers along time:
//
//     uint64 pl_mask_exp  [ringbuf_size_in_64] [num_frequencies_in] [num_elements]   (input)
//     uint64 pl_mask_exp_U[ringbuf_size_out_64][num_frequencies_out][num_elements]   (output)
//
// This processes input frequency channels [Fmin, Fmax) and writes them to output channels
// [0, Fmax-Fmin). The output buffer may over-allocate frequencies (num_frequencies_out >= Fmax-Fmin);
// the unwritten channels are left untouched. Logical input row `r` lives at physical row
// `(ringbuf_pos_in_64 + r) % ringbuf_size_in_64`; likewise for the output.
//
//   pl_mask_exp_U      - output buffer (upchannelized mask)
//   pl_mask_exp        - input buffer (expanded mask)
//   num_elements       - inner/fast dimension (= num_polarizations * num_dishes/8)
//   num_frequencies_in - input frequency dimension
//   num_frequencies_out- output frequency dimension (total available; >= Fmax - Fmin)
//   Fmin, Fmax         - input coarse-channel range to process; output is written at [0, Fmax-Fmin)
//   num_times_64       - number of input `time_64` elements to consume (produces num_times_64/U)
//   ringbuf_size_in_64 - input  ring buffer size in input `time_64`; guaranteed a power of two
//   ringbuf_pos_in_64  - input  ring buffer current start position in input `time_64`
//   ringbuf_size_out_64- output ring buffer size in output `time_64` (uint64 elements, each
//                        covering 64*U input time samples); guaranteed a power of two
//   ringbuf_pos_out_64 - output ring buffer current start position in output `time_64`
//   U                  - upchannelization factor, a power of two in {2,4,8,16,32,64,128}
//   stream             - CUDA stream to launch on
void launch_upchannelize_pl_mask(std::uint64_t* pl_mask_exp_U, const std::uint64_t* pl_mask_exp,
                                 std::ptrdiff_t num_elements, std::ptrdiff_t num_frequencies_in,
                                 std::ptrdiff_t num_frequencies_out, int Fmin, int Fmax,
                                 std::ptrdiff_t num_times_64, std::ptrdiff_t ringbuf_size_in_64,
                                 std::ptrdiff_t ringbuf_pos_in_64, std::ptrdiff_t ringbuf_size_out_64,
                                 std::ptrdiff_t ringbuf_pos_out_64, int U, cudaStream_t stream);

// CPU reference for `launch_upchannelize_pl_mask` (same arguments, minus the CUDA stream). This is
// the slow, straightforward, "obviously correct" version used to validate the GPU kernel; if the GPU
// kernel disagrees with it, the GPU kernel is wrong.
void cpu_upchannelize_pl_mask(std::uint64_t* pl_mask_exp_U, const std::uint64_t* pl_mask_exp,
                              std::ptrdiff_t num_elements, std::ptrdiff_t num_frequencies_in,
                              std::ptrdiff_t num_frequencies_out, int Fmin, int Fmax,
                              std::ptrdiff_t num_times_64, std::ptrdiff_t ringbuf_size_in_64,
                              std::ptrdiff_t ringbuf_pos_in_64, std::ptrdiff_t ringbuf_size_out_64,
                              std::ptrdiff_t ringbuf_pos_out_64, int U);

#endif // CUDA_PL_MASK_UPCHANNELIZER_HPP
