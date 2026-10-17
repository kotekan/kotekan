// Packet-loss (PL) mask kernel for the FRB1 output.
//
// See cudaPLMaskFRB1.hpp for the public launcher API.

#include "cudaPLMaskFRB1.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Helpers

namespace {

// Mask selecting the bits [lo, hi) of a 64-bit word, with 0 <= lo < hi <= 64
__host__ __device__ inline std::uint64_t bit_range(const int lo, const int hi) {
    const std::uint64_t below_hi = hi == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << hi) - 1;
    return below_hi & ~((std::uint64_t(1) << lo) - 1);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Kernel

// One block per output sample `k` (blockIdx.x) and input frequency `Fmin + blockIdx.y`. The threads
// loop over the elements, AND the input bits of the time window
// [bit_offset_in + Td * k, bit_offset_in + Td * (k + 1)), and then combine their results.
__global__ void
frb1_pl_mask(std::uint8_t* __restrict__ const frb_pl_mask,
             const std::uint64_t* __restrict__ const pl_mask, const std::ptrdiff_t num_elements,
             const std::ptrdiff_t num_frequencies_in, const std::ptrdiff_t num_frequencies_out,
             const int Fmin, const int Fbar_out_min, const int U, const int Td,
             const std::ptrdiff_t ringbuf_size_in_64, const std::ptrdiff_t ringbuf_pos_in_64,
             const int bit_offset_in, const std::ptrdiff_t ringbuf_size_out,
             const std::ptrdiff_t ringbuf_pos_out) {
    const std::ptrdiff_t k = blockIdx.x;
    const int f = Fmin + int(blockIdx.y);

    const std::ptrdiff_t t_begin = bit_offset_in + Td * k;
    const std::ptrdiff_t t_end = t_begin + Td;

    bool valid = true;
    for (std::ptrdiff_t e = threadIdx.x; e < num_elements; e += blockDim.x) {
        for (std::ptrdiff_t r = t_begin / 64; r <= (t_end - 1) / 64; ++r) {
            const int lo = r == t_begin / 64 ? int(t_begin % 64) : 0;
            const int hi = r == (t_end - 1) / 64 ? int((t_end - 1) % 64) + 1 : 64;
            const std::uint64_t bits = bit_range(lo, hi);
            const std::ptrdiff_t phys = (ringbuf_pos_in_64 + r) % ringbuf_size_in_64;
            const std::uint64_t word = pl_mask[(phys * num_frequencies_in + f) * num_elements + e];
            valid &= (word & bits) == bits;
        }
    }
    valid = __syncthreads_and(valid);

    const std::ptrdiff_t phys_out = (ringbuf_pos_out + k) % ringbuf_size_out;
    std::uint8_t* const out_row = frb_pl_mask + phys_out * num_frequencies_out + Fbar_out_min
                                  + std::ptrdiff_t(U) * (f - Fmin);
    for (int u = threadIdx.x; u < U; u += blockDim.x)
        out_row[u] = valid;
}

////////////////////////////////////////////////////////////////////////////////
// Launcher (externally visible)

void launch_frb1_pl_mask(std::uint8_t* const frb_pl_mask, const std::uint64_t* const pl_mask,
                         const std::ptrdiff_t num_elements, const std::ptrdiff_t num_frequencies_in,
                         const std::ptrdiff_t num_frequencies_out, const int Fmin, const int Fmax,
                         const int Fbar_out_min, const int U, const int Td,
                         const std::ptrdiff_t num_times_out,
                         const std::ptrdiff_t ringbuf_size_in_64,
                         const std::ptrdiff_t ringbuf_pos_in_64, const int bit_offset_in,
                         const std::ptrdiff_t ringbuf_size_out,
                         const std::ptrdiff_t ringbuf_pos_out, const cudaStream_t stream) {
    assert(frb_pl_mask);
    assert(pl_mask);
    assert(num_elements > 0);
    assert(0 <= Fmin && Fmin <= Fmax && Fmax <= num_frequencies_in);
    assert(U >= 1 && Td >= 1);
    assert(0 <= Fbar_out_min && Fbar_out_min + U * (Fmax - Fmin) <= num_frequencies_out);
    assert(num_times_out >= 0);
    assert(ringbuf_size_in_64 > 0 && ringbuf_pos_in_64 >= 0);
    assert(0 <= bit_offset_in && bit_offset_in < 64);
    assert(ringbuf_size_out > 0 && ringbuf_pos_out >= 0);
    assert(num_times_out <= ringbuf_size_out);

    if (num_times_out == 0 || Fmax == Fmin)
        return;

    constexpr int threads_x = 128;
    const dim3 nthreads(threads_x, 1, 1);
    const dim3 nblocks(num_times_out, Fmax - Fmin, 1);
    frb1_pl_mask<<<nblocks, nthreads, 0, stream>>>(
        frb_pl_mask, pl_mask, num_elements, num_frequencies_in, num_frequencies_out, Fmin,
        Fbar_out_min, U, Td, ringbuf_size_in_64, ringbuf_pos_in_64, bit_offset_in, ringbuf_size_out,
        ringbuf_pos_out);
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference (the slow, "obviously correct" version; the oracle for the kernel)

void cpu_frb1_pl_mask(std::uint8_t* const frb_pl_mask, const std::uint64_t* const pl_mask,
                      const std::ptrdiff_t num_elements, const std::ptrdiff_t num_frequencies_in,
                      const std::ptrdiff_t num_frequencies_out, const int Fmin, const int Fmax,
                      const int Fbar_out_min, const int U, const int Td,
                      const std::ptrdiff_t num_times_out, const std::ptrdiff_t ringbuf_size_in_64,
                      const std::ptrdiff_t ringbuf_pos_in_64, const int bit_offset_in,
                      const std::ptrdiff_t ringbuf_size_out, const std::ptrdiff_t ringbuf_pos_out) {
    assert(frb_pl_mask);
    assert(pl_mask);
    assert(num_elements > 0);
    assert(0 <= Fmin && Fmin <= Fmax && Fmax <= num_frequencies_in);
    assert(U >= 1 && Td >= 1);
    assert(0 <= Fbar_out_min && Fbar_out_min + U * (Fmax - Fmin) <= num_frequencies_out);
    assert(num_times_out >= 0);
    assert(ringbuf_size_in_64 > 0 && ringbuf_pos_in_64 >= 0);
    assert(0 <= bit_offset_in && bit_offset_in < 64);
    assert(ringbuf_size_out > 0 && ringbuf_pos_out >= 0);
    assert(num_times_out <= ringbuf_size_out);

    // Read input bit `t` (logical time) of frequency `f` and element `e`, with ring wrap
    const auto in_bit = [&](const std::ptrdiff_t t, const int f, const std::ptrdiff_t e) {
        const std::ptrdiff_t phys = (ringbuf_pos_in_64 + t / 64) % ringbuf_size_in_64;
        return (pl_mask[(phys * num_frequencies_in + f) * num_elements + e] >> (t % 64)) & 1;
    };

    for (std::ptrdiff_t k = 0; k < num_times_out; ++k) {
        for (int f = Fmin; f < Fmax; ++f) {
            bool valid = true;
            for (std::ptrdiff_t e = 0; e < num_elements; ++e)
                for (std::ptrdiff_t t = bit_offset_in + Td * k; t < bit_offset_in + Td * (k + 1);
                     ++t)
                    valid &= in_bit(t, f, e) != 0;
            for (int u = 0; u < U; ++u)
                frb_pl_mask[(ringbuf_pos_out + k) % ringbuf_size_out * num_frequencies_out
                            + Fbar_out_min + std::ptrdiff_t(U) * (f - Fmin) + u] = valid;
        }
    }
}
