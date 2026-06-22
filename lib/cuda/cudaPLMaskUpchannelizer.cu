// Packet-loss (PL) mask upchannelizer kernel for an A40 GPU (sm_86).
//
// See cudaPLMaskUpchannelizer.hpp for the public launcher API.

#include "cudaPLMaskUpchannelizer.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Free functions (no namespace) so the cudaPLMaskUpchannelizer cudaCommand class can use that name;
// mirrors the gpu_/cpu_ pairing in cudaQuantizeKernel8.cu.

////////////////////////////////////////////////////////////////////////////////
// Helpers

// Internal to this translation unit.
namespace {

// AND-reduce each contiguous group of `U` bits within a 64-bit word.
//
// After the fold, bit position `p` of the result holds the logical AND of the
// original bits [p, p+U). In particular the AND of group `g` (original bits
// [g*U, g*U+U)) sits at bit position `g*U`. `U` must be a power of two; only the
// `U < 64` case is used (see the kernel).
template<int U>
__device__ inline std::uint64_t and_downsample_word(std::uint64_t w) {
#pragma unroll
    for (int sh = 1; sh < U; sh <<= 1)
        w &= w >> sh;
    return w;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Kernel

// Upchannelize the expanded PL mask by a factor `U`.
//
// Array layouts:
//     uint64 pl_mask_exp  [ringbuf_size_in_64] [num_spectators]   (input)
//     uint64 pl_mask_exp_U[ringbuf_size_out_64][num_spectators]   (output)
//
// Each array is addressed bit-wise along time. The time index is split into two
// parts: the low six bits (time % 64) live within the `uint64` array elements,
// and the high bits (time / 64) index the (ring-buffered) first dimension.
//
// The algorithm ANDs M*U input bits into one output bit; it downsamples U input
// bits to one output bit, with an implicit forward stencil of (M-1)*U bits:
//
//     output bit `tbar`  =  AND of input bits [tbar*U, tbar*U + M*U)
//
// It is computed as a clean two-step reduction:
//     A[k]        = AND of input bits [k*U, (k+1)*U)          (downsample-by-U)
//     out[tbar]   = A[tbar] & A[tbar+1] & ... & A[tbar+M-1]   (sliding M-tap AND)
//
// The input and output are independent power-of-two ring buffers along time:
//   - input ring  is measured in input `time_64`  (uint64 elements; 64 time bits each);
//     logical input row `r`  is at physical row `(ringbuf_pos_in_64 + r) & (ringbuf_size_in_64 - 1)`.
//   - output ring is measured in output `time_64` (uint64 elements; each output element
//     covers 64*U input time samples); logical output row `o` is at physical row
//     `(ringbuf_pos_out_64 + o) & (ringbuf_size_out_64 - 1)`.
// Reading the (M-1)*U future bits past the consumed range is safe -- those
// elements are guaranteed readable (they wrap within the input ring).
//
// Parallelization: each thread handles one spectator `s` (the inner, fast index,
// so consecutive threads touch consecutive addresses -> fully coalesced) and a
// contiguous run of `N` output words. Producing N words at once amortizes the
// (M-1)*U stencil overhang: it is re-read once per N-word run rather than once
// per output word, driving the read amplification to ~1.
template<int M, int U, int N>
__global__ void upchannelize_pl_mask(std::uint64_t* __restrict__ const pl_mask_exp_U,
                                     const std::uint64_t* __restrict__ const pl_mask_exp,
                                     const std::ptrdiff_t num_spectators,
                                     const std::ptrdiff_t num_times_64,
                                     const std::ptrdiff_t ringbuf_size_in_64,
                                     const std::ptrdiff_t ringbuf_pos_in_64,
                                     const std::ptrdiff_t ringbuf_size_out_64,
                                     const std::ptrdiff_t ringbuf_pos_out_64) {
    static_assert(M >= 1, "M (PFB taps) must be positive");
    static_assert(U >= 2 && (U & (U - 1)) == 0, "U must be a power of two >= 2");
    static_assert(N >= 1, "N (output words per thread) must be positive");

    // Spectator handled by this thread (inner/fast index -> coalesced accesses).
    const std::ptrdiff_t s = std::ptrdiff_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= num_spectators)
        return;

    // Independent input/output ring geometries (sizes are powers of two).
    const std::ptrdiff_t num_out_words = num_times_64 / U; // output uint64 rows produced
    const std::ptrdiff_t in_mask = ringbuf_size_in_64 - 1; // power-of-two wrap masks
    const std::ptrdiff_t out_mask = ringbuf_size_out_64 - 1;

    // This thread owns the contiguous output-word run [T0, T0 + n_words).
    const std::ptrdiff_t T0 = std::ptrdiff_t(blockIdx.y) * N;
    if (T0 >= num_out_words)
        return;
    const int n_words = T0 + N <= num_out_words ? N : int(num_out_words - T0);

    // First input row to read, and number of A-bits to produce. The trailing
    // (M-1) A-bits feed the forward stencil of the last output bit.
    const std::ptrdiff_t r_lo = T0 * std::ptrdiff_t(U);
    const int n_abits = n_words * 64 + (M - 1);

    // Sliding M-tap AND window (circular over the last M A-bits) + output packer.
    std::uint64_t awin[M];
    int n = 0;                  // count of A-bits produced so far
    std::uint64_t out_word = 0; // output word being assembled
    int out_bit = 0;            // next bit position within out_word (0..63)
    std::ptrdiff_t out_row = 0; // output words written so far (0..n_words)

    // Push one A-bit through the sliding AND and pack/emit output bits.
    const auto push_abit = [&](const std::uint64_t abit) {
        awin[n % M] = abit;
        if (n >= M - 1) {
            std::uint64_t outbit = 1;
#pragma unroll
            for (int m = 0; m < M; ++m)
                outbit &= awin[m];
            out_word |= outbit << out_bit;
            if (++out_bit == 64) {
                const std::ptrdiff_t phys = (ringbuf_pos_out_64 + T0 + out_row) & out_mask;
                pl_mask_exp_U[phys * num_spectators + s] = out_word;
                out_word = 0;
                out_bit = 0;
                ++out_row;
            }
        }
        ++n;
    };

    if constexpr (U < 64) {
        // Each input word yields `groups_per_word` A-bits. Row r_lo's first group
        // aligns to k0 = T0*64, so iterating rows yields A-bits in increasing k.
        constexpr int groups_per_word = 64 / U;
        std::ptrdiff_t r = r_lo;
        while (n < n_abits) {
            const std::ptrdiff_t phys = (ringbuf_pos_in_64 + r) & in_mask;
            const std::uint64_t w = and_downsample_word<U>(pl_mask_exp[phys * num_spectators + s]);
            const int take = groups_per_word < n_abits - n ? groups_per_word : n_abits - n;
            for (int g = 0; g < take; ++g)
                push_abit((w >> (g * U)) & std::uint64_t(1));
            ++r;
        }
    } else {
        // U >= 64: each group spans U/64 whole input words; A[k] = 1 iff all of
        // them are all-ones.
        constexpr int words_per_group = U / 64;
        std::ptrdiff_t r = r_lo;
        while (n < n_abits) {
            std::uint64_t acc = ~std::uint64_t(0);
#pragma unroll
            for (int j = 0; j < words_per_group; ++j) {
                const std::ptrdiff_t phys = (ringbuf_pos_in_64 + r) & in_mask;
                acc &= pl_mask_exp[phys * num_spectators + s];
                ++r;
            }
            push_abit(acc == ~std::uint64_t(0) ? std::uint64_t(1) : std::uint64_t(0));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Launcher (externally visible)

void launch_upchannelize_pl_mask(std::uint64_t* const pl_mask_exp_U,
                                 const std::uint64_t* const pl_mask_exp,
                                 const std::ptrdiff_t num_spectators,
                                 const std::ptrdiff_t num_times_64,
                                 const std::ptrdiff_t ringbuf_size_in_64,
                                 const std::ptrdiff_t ringbuf_pos_in_64,
                                 const std::ptrdiff_t ringbuf_size_out_64,
                                 const std::ptrdiff_t ringbuf_pos_out_64, const int U,
                                 const cudaStream_t stream) {
    assert(pl_mask_exp_U);
    assert(pl_mask_exp);
    assert(num_spectators > 0);
    assert(num_times_64 > 0 && num_times_64 % U == 0);
    // Both ring buffer sizes must be powers of two.
    assert(ringbuf_size_in_64 > 0 && (ringbuf_size_in_64 & (ringbuf_size_in_64 - 1)) == 0);
    assert(ringbuf_size_out_64 > 0 && (ringbuf_size_out_64 & (ringbuf_size_out_64 - 1)) == 0);
    assert(ringbuf_pos_in_64 >= 0);
    assert(ringbuf_pos_out_64 >= 0);

    // M = number of PFB taps; N = output words produced per thread.
    constexpr int M = 4;
    constexpr int N = 256;
    constexpr int threads_x = 256;

    const std::ptrdiff_t num_out_words = num_times_64 / U;
    if (num_out_words <= 0)
        return;

    const dim3 nthreads(threads_x, 1, 1);
    const dim3 nblocks((num_spectators + threads_x - 1) / threads_x, (num_out_words + N - 1) / N, 1);

#define LAUNCH_UPCHAN_PL_MASK(UU)                                                                  \
    upchannelize_pl_mask<M, (UU), N><<<nblocks, nthreads, 0, stream>>>(                            \
        pl_mask_exp_U, pl_mask_exp, num_spectators, num_times_64, ringbuf_size_in_64,              \
        ringbuf_pos_in_64, ringbuf_size_out_64, ringbuf_pos_out_64)

    switch (U) {
        case 2:
            LAUNCH_UPCHAN_PL_MASK(2);
            break;
        case 4:
            LAUNCH_UPCHAN_PL_MASK(4);
            break;
        case 8:
            LAUNCH_UPCHAN_PL_MASK(8);
            break;
        case 16:
            LAUNCH_UPCHAN_PL_MASK(16);
            break;
        case 32:
            LAUNCH_UPCHAN_PL_MASK(32);
            break;
        case 64:
            LAUNCH_UPCHAN_PL_MASK(64);
            break;
        case 128:
            LAUNCH_UPCHAN_PL_MASK(128);
            break;
        default:
            assert(false && "U must be a power of two in {2,4,8,16,32,64,128}");
    }

#undef LAUNCH_UPCHAN_PL_MASK
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference (the slow, "obviously correct" version; the oracle for the kernel)

void cpu_upchannelize_pl_mask(std::uint64_t* const pl_mask_exp_U,
                              const std::uint64_t* const pl_mask_exp,
                              const std::ptrdiff_t num_spectators, const std::ptrdiff_t num_times_64,
                              const std::ptrdiff_t ringbuf_size_in_64,
                              const std::ptrdiff_t ringbuf_pos_in_64,
                              const std::ptrdiff_t ringbuf_size_out_64,
                              const std::ptrdiff_t ringbuf_pos_out_64, const int U) {
    assert(pl_mask_exp_U);
    assert(pl_mask_exp);
    // Both ring buffer sizes must be powers of two; num_times_64 must be a multiple of U (matches
    // the GPU kernel contract).
    assert(ringbuf_size_in_64 > 0 && (ringbuf_size_in_64 & (ringbuf_size_in_64 - 1)) == 0);
    assert(ringbuf_size_out_64 > 0 && (ringbuf_size_out_64 & (ringbuf_size_out_64 - 1)) == 0);
    assert(ringbuf_pos_in_64 >= 0);
    assert(ringbuf_pos_out_64 >= 0);
    assert(num_times_64 % U == 0);

    constexpr int M = 4; // PFB taps

    const std::ptrdiff_t num_out_words = num_times_64 / U;  // output uint64 rows produced
    const std::ptrdiff_t in_mask = ringbuf_size_in_64 - 1;  // power-of-two wrap masks
    const std::ptrdiff_t out_mask = ringbuf_size_out_64 - 1;

    // Read input bit `t` (logical time) for spectator `s`, with ring wrap.
    const auto in_bit = [&](const std::ptrdiff_t t, const std::ptrdiff_t s) -> std::uint64_t {
        const std::ptrdiff_t r = t >> 6; // logical input row = t / 64
        const int b = int(t & 63);       // bit within the word = t % 64
        const std::ptrdiff_t phys = (ringbuf_pos_in_64 + r) & in_mask;
        return (pl_mask_exp[phys * num_spectators + s] >> b) & std::uint64_t(1);
    };

    for (std::ptrdiff_t o = 0; o < num_out_words; ++o) {
        const std::ptrdiff_t phys_out = (ringbuf_pos_out_64 + o) & out_mask;
        for (std::ptrdiff_t s = 0; s < num_spectators; ++s) {
            std::uint64_t out_word = 0;
            for (int j = 0; j < 64; ++j) {
                const std::ptrdiff_t tbar = o * 64 + j;
                // Output bit = AND of the M*U input bits [tbar*U, tbar*U + M*U).
                std::uint64_t bit = 1;
                const std::ptrdiff_t t0 = tbar * std::ptrdiff_t(U);
                for (std::ptrdiff_t t = t0; t < t0 + std::ptrdiff_t(M) * U; ++t)
                    bit &= in_bit(t, s);
                out_word |= bit << j;
            }
            pl_mask_exp_U[phys_out * num_spectators + s] = out_word;
        }
    }
}
