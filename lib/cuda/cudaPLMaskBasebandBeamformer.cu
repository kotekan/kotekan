// Packet-loss (PL) mask for the baseband beamformer outputs.
//
// See cudaPLMaskBasebandBeamformer.hpp for the public launcher API.

#include "cudaPLMaskBasebandBeamformer.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// One thread per output word: AND the `num_d8` input words of the same row and inner index.
__global__ void bb_pl_mask_kernel(std::uint64_t* __restrict__ const bb_pl_mask,
                                  const std::uint64_t* __restrict__ const pl_mask_exp,
                                  const std::ptrdiff_t num_inner, const std::ptrdiff_t num_d8,
                                  const std::ptrdiff_t num_rows,
                                  const std::ptrdiff_t ringbuf_size_in,
                                  const std::ptrdiff_t ringbuf_pos_in,
                                  const std::ptrdiff_t ringbuf_size_out,
                                  const std::ptrdiff_t ringbuf_pos_out) {
    const std::ptrdiff_t s = std::ptrdiff_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::ptrdiff_t r = blockIdx.y;
    if (s >= num_inner || r >= num_rows)
        return;

    const std::ptrdiff_t row_in = (ringbuf_pos_in + r) % ringbuf_size_in;
    const std::ptrdiff_t row_out = (ringbuf_pos_out + r) % ringbuf_size_out;
    const std::uint64_t* const in = &pl_mask_exp[(row_in * num_inner + s) * num_d8];
    std::uint64_t acc = ~std::uint64_t(0);
    for (std::ptrdiff_t d8 = 0; d8 < num_d8; ++d8)
        acc &= in[d8];
    bb_pl_mask[row_out * num_inner + s] = acc;
}

void launch_bb_pl_mask(std::uint64_t* const bb_pl_mask, const std::uint64_t* const pl_mask_exp,
                       const std::ptrdiff_t num_inner, const std::ptrdiff_t num_d8,
                       const std::ptrdiff_t num_rows, const std::ptrdiff_t ringbuf_size_in,
                       const std::ptrdiff_t ringbuf_pos_in, const std::ptrdiff_t ringbuf_size_out,
                       const std::ptrdiff_t ringbuf_pos_out, const cudaStream_t stream) {
    assert(bb_pl_mask);
    assert(pl_mask_exp);
    assert(num_inner > 0 && num_d8 > 0 && num_rows >= 0);
    assert(ringbuf_size_in > 0 && 0 <= ringbuf_pos_in);
    assert(ringbuf_size_out > 0 && 0 <= ringbuf_pos_out);
    assert(num_rows <= ringbuf_size_in && num_rows <= ringbuf_size_out);
    assert(num_rows <= 65535); // gridDim.y limit

    if (num_rows == 0)
        return;

    constexpr int threads_x = 256;
    const dim3 nthreads(threads_x, 1, 1);
    const dim3 nblocks((num_inner + threads_x - 1) / threads_x, num_rows, 1);
    bb_pl_mask_kernel<<<nblocks, nthreads, 0, stream>>>(bb_pl_mask, pl_mask_exp, num_inner, num_d8,
                                                        num_rows, ringbuf_size_in, ringbuf_pos_in,
                                                        ringbuf_size_out, ringbuf_pos_out);
}

void cpu_bb_pl_mask(std::uint64_t* const bb_pl_mask, const std::uint64_t* const pl_mask_exp,
                    const std::ptrdiff_t num_inner, const std::ptrdiff_t num_d8,
                    const std::ptrdiff_t num_rows, const std::ptrdiff_t ringbuf_size_in,
                    const std::ptrdiff_t ringbuf_pos_in, const std::ptrdiff_t ringbuf_size_out,
                    const std::ptrdiff_t ringbuf_pos_out) {
    assert(bb_pl_mask);
    assert(pl_mask_exp);
    assert(num_inner > 0 && num_d8 > 0 && num_rows >= 0);
    assert(ringbuf_size_in > 0 && 0 <= ringbuf_pos_in);
    assert(ringbuf_size_out > 0 && 0 <= ringbuf_pos_out);
    assert(num_rows <= ringbuf_size_in && num_rows <= ringbuf_size_out);

    for (std::ptrdiff_t r = 0; r < num_rows; ++r) {
        const std::ptrdiff_t row_in = (ringbuf_pos_in + r) % ringbuf_size_in;
        const std::ptrdiff_t row_out = (ringbuf_pos_out + r) % ringbuf_size_out;
        for (std::ptrdiff_t s = 0; s < num_inner; ++s) {
            std::uint64_t acc = ~std::uint64_t(0);
            for (std::ptrdiff_t d8 = 0; d8 < num_d8; ++d8)
                acc &= pl_mask_exp[(row_in * num_inner + s) * num_d8 + d8];
            bb_pl_mask[row_out * num_inner + s] = acc;
        }
    }
}
