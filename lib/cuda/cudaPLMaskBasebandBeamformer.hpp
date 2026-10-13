// Packet-loss (PL) mask for the baseband beamformer outputs.
//
// Every dish contributes to every baseband beam, so a beam sample at (time, frequency,
// polarization) is valid only if all dishes are valid there: the logical AND of the expanded PL
// mask over its dish axis. The result is the same for all beams.

#ifndef CUDA_PL_MASK_BASEBAND_BEAMFORMER_HPP
#define CUDA_PL_MASK_BASEBAND_BEAMFORMER_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Array layouts (each `uint64` holds 64 consecutive time bits). Input and output are ring buffers
// along time, measured in `time_64` rows; logical row `r` lives at physical row
// `(ringbuf_pos + r) % ringbuf_size`:
//
//     uint64 pl_mask_exp[ringbuf_size_in] [num_inner][num_d8]   (input)
//     uint64 bb_pl_mask [ringbuf_size_out][num_inner]           (output)
//
//   bb_pl_mask       - output buffer
//   pl_mask_exp      - input buffer (expanded PL mask)
//   num_inner        - number of output words per row (= num_frequencies * num_polarizations)
//   num_d8           - number of dish groups ANDed into one output word (= num_dishes / 8)
//   num_rows         - number of `time_64` rows to process
//   ringbuf_size_in  - input ring buffer size in rows
//   ringbuf_pos_in   - input ring buffer start position in rows
//   ringbuf_size_out - output ring buffer size in rows
//   ringbuf_pos_out  - output ring buffer start position in rows
//   stream           - CUDA stream to launch on
void launch_bb_pl_mask(std::uint64_t* bb_pl_mask, const std::uint64_t* pl_mask_exp,
                       std::ptrdiff_t num_inner, std::ptrdiff_t num_d8, std::ptrdiff_t num_rows,
                       std::ptrdiff_t ringbuf_size_in, std::ptrdiff_t ringbuf_pos_in,
                       std::ptrdiff_t ringbuf_size_out, std::ptrdiff_t ringbuf_pos_out,
                       cudaStream_t stream);

// CPU reference for `launch_bb_pl_mask` (same arguments, minus the CUDA stream).
void cpu_bb_pl_mask(std::uint64_t* bb_pl_mask, const std::uint64_t* pl_mask_exp,
                    std::ptrdiff_t num_inner, std::ptrdiff_t num_d8, std::ptrdiff_t num_rows,
                    std::ptrdiff_t ringbuf_size_in, std::ptrdiff_t ringbuf_pos_in,
                    std::ptrdiff_t ringbuf_size_out, std::ptrdiff_t ringbuf_pos_out);

#endif // CUDA_PL_MASK_BASEBAND_BEAMFORMER_HPP
