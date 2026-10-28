// Packet-loss (PL) mask kernel for the FRB1 output.
//
// Downsamples an expanded PL mask (upchannelized or not) in time, exactly as the FRB1 beamformer
// downsamples its input voltages: FRB1 output sample `ttilde` sums the `Td` input samples
// [Td * ttilde, Td * ttilde + Td) over all dishes and polarizations, so it is valid only if all of
// those input bits are valid -- a logical AND.

#ifndef CUDA_PL_MASK_FRB1_HPP
#define CUDA_PL_MASK_FRB1_HPP

#include <cstddef>        // for ptrdiff_t
#include <cstdint>        // for uint64_t, uint8_t
#include <driver_types.h> // for cudaStream_t

// These are free functions (not in a namespace) so the `cudaPLMaskFRB1` cudaCommand class can use
// that name; this mirrors the gpu_/cpu_ pairing in cudaQuantizeKernel8.hpp.

// Downsample the expanded PL mask by `Td` in time, AND over all elements, and write one byte per
// output sample and output frequency.
//
// Array layouts:
//
//     uint64 pl_mask    [ringbuf_size_in_64][num_frequencies_in][num_elements]   (input)
//     uint8  frb_pl_mask[ringbuf_size_out][num_frequencies_out]                  (output)
//
// Each input `uint64` holds 64 consecutive time bits. Logical input row `r` lives at physical row
// `(ringbuf_pos_in_64 + r) % ringbuf_size_in_64`, and the input time axis starts `bit_offset_in`
// bits into logical row 0. Output row `k` lives at physical row
// `(ringbuf_pos_out + k) % ringbuf_size_out`. Output byte `(k, Fbar_out_min + U * (f - Fmin) +
// u)` is 1 iff all input bits [bit_offset_in + Td * k, bit_offset_in + Td * (k + 1)) of input
// frequency `f` are 1 for all elements, for all `u` in [0, U). Other output bytes are left
// untouched.
//
//   frb_pl_mask         - output buffer
//   pl_mask             - input buffer
//   num_elements        - inner/fast input dimension (= num_polarizations * num_dishes/8)
//   num_frequencies_in  - input frequency dimension
//   num_frequencies_out - output frequency dimension
//   Fmin, Fmax          - input frequency range to process
//   Fbar_out_min        - first output frequency to write
//   U                   - upchannelization factor of the input; each input frequency is written
//                         to U consecutive output frequencies
//   Td                  - time downsampling factor (input samples per output sample)
//   num_times_out       - number of output samples to produce
//   ringbuf_size_in_64  - input ring buffer size in `time_64` rows
//   ringbuf_pos_in_64   - input ring buffer start position in `time_64` rows
//   bit_offset_in       - first input time sample within the first input row, in [0, 64)
//   ringbuf_size_out    - output ring buffer size in output samples
//   ringbuf_pos_out     - output ring buffer start position in output samples
//   stream              - CUDA stream to launch on
void launch_frb1_pl_mask(std::uint8_t* frb_pl_mask, const std::uint64_t* pl_mask,
                         std::ptrdiff_t num_elements, std::ptrdiff_t num_frequencies_in,
                         std::ptrdiff_t num_frequencies_out, int Fmin, int Fmax, int Fbar_out_min,
                         int U, int Td, std::ptrdiff_t num_times_out,
                         std::ptrdiff_t ringbuf_size_in_64, std::ptrdiff_t ringbuf_pos_in_64,
                         int bit_offset_in, std::ptrdiff_t ringbuf_size_out,
                         std::ptrdiff_t ringbuf_pos_out, cudaStream_t stream);

// CPU reference for `launch_frb1_pl_mask` (same arguments, minus the CUDA stream). This is the
// slow, straightforward, "obviously correct" version used to validate the GPU kernel; if the GPU
// kernel disagrees with it, the GPU kernel is wrong.
void cpu_frb1_pl_mask(std::uint8_t* frb_pl_mask, const std::uint64_t* pl_mask,
                      std::ptrdiff_t num_elements, std::ptrdiff_t num_frequencies_in,
                      std::ptrdiff_t num_frequencies_out, int Fmin, int Fmax, int Fbar_out_min,
                      int U, int Td, std::ptrdiff_t num_times_out,
                      std::ptrdiff_t ringbuf_size_in_64, std::ptrdiff_t ringbuf_pos_in_64,
                      int bit_offset_in, std::ptrdiff_t ringbuf_size_out,
                      std::ptrdiff_t ringbuf_pos_out);

#endif // CUDA_PL_MASK_FRB1_HPP
