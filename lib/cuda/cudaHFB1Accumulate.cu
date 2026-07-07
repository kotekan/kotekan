// Accumulate HFB 1 beams (consuming output of the FRB1 stage, producing input for the FRB2 stage)
//
// See cudaHFB1Accumulate.hpp for the public launcher API.

#include "cudaHFB1Accumulate.hpp"

#include <cassert>
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

    // CHIME dish layout
    constexpr int P = 2 * 256;
    constexpr int Q = 2 * 4;

    // A struct holding 4 consecutive `half2` values (8 `__half` values = 128 bits). This mirrors
    // the vectorized load idiom in cudaQuantizeKernel8.cu: loading/storing all 8 halves with a
    // single 128-bit (`int4`) instruction is more efficient than touching each `half2` separately.
    struct half2x4 {
        __half2 x, y, z, w;
    };

    __device__ inline half2x4 load_half2x4(const __half* __restrict__ const p) {
        const int4 d = *reinterpret_cast<const int4*>(p);
        return reinterpret_cast<const half2x4&>(d);
    }

    __device__ inline void store_half2x4(__half* __restrict__ const p, const half2x4 v) {
        *reinterpret_cast<int4*>(p) = reinterpret_cast<const int4&>(v);
    }

} // namespace

__global__ void accumulate_hfb1(__half* __restrict__ const accumulator,
                                const __half* __restrict__ const input, const int num_frequencies,
                                const int num_times) {
  // Array indexing:
  //     input[T][F][PQ]
  //     accumulator[F][PQ]
  const int frequency = blockIdx.x;
  const int pq = 8 * threadIdx.x;

  half2x4 accum;
  accum.x = accum.y = accum.z = accum.w = __float2half2_rn(0.0f);
  for (int time = 0; time < num_times; ++time) {
    const std::ptrdiff_t input_offset = pq + P*Q * frequency + P*Q * num_frequencies * time;
    // load 8 __half values per thread from input + offset (single 128-bit load) and add them
    const half2x4 in = load_half2x4(input + input_offset);
    accum.x = __hadd2(accum.x, in.x);
    accum.y = __hadd2(accum.y, in.y);
    accum.z = __hadd2(accum.z, in.z);
    accum.w = __hadd2(accum.w, in.w);
  }

  // Divide by num_times (time average): multiply once by the broadcast reciprocal.
  const __half2 scale = __half2half2(__float2half(1.0f / num_times));
  accum.x = __hmul2(accum.x, scale);
  accum.y = __hmul2(accum.y, scale);
  accum.z = __hmul2(accum.z, scale);
  accum.w = __hmul2(accum.w, scale);

  const std::ptrdiff_t accumulator_offset = pq + P*Q * frequency;
  // store 8 __half values per thread to accumulator + offset (single 128-bit store)
  store_half2x4(accumulator + accumulator_offset, accum);
}

////////////////////////////////////////////////////////////////////////////////
// Launcher (externally visible)

void launch_accumulate_hfb1(__half* const accumulator, const __half* const input,
                            const int num_frequencies, const int num_times,
                            const cudaStream_t stream) {
    assert(accumulator);
    assert(input);
    assert(num_frequencies > 0);
    assert(num_times > 0);

    // Each thread owns 8 contiguous halves; one block handles one frequency's P*Q halves.
    const dim3 nthreads(P * Q / 8, 1, 1); // 512 threads = 4096 halves / 8
    const dim3 nblocks(num_frequencies, 1, 1);

    accumulate_hfb1<<<nblocks, nthreads, 0, stream>>>(accumulator, input, num_frequencies,
                                                      num_times);
}
