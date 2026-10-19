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

    __device__ inline float2 add_half2(const float2 a, const __half2 b) {
        const float2 c = __half22float2(b);
        return make_float2(a.x + c.x, a.y + c.y);
    }

    __device__ inline __half2 scale_to_half2(const float2 a, const float scale) {
        return __floats2half2_rn(a.x * scale, a.y * scale);
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

  // Sum in float: the sum of num_times samples can exceed the float16 range even when their
  // average does not.
  float2 sum_x = make_float2(0.0f, 0.0f), sum_y = sum_x, sum_z = sum_x, sum_w = sum_x;
  for (int time = 0; time < num_times; ++time) {
    const std::ptrdiff_t input_offset = pq + P*Q * frequency + P*Q * num_frequencies * time;
    // load 8 __half values per thread from input + offset (single 128-bit load) and add them
    const half2x4 in = load_half2x4(input + input_offset);
    sum_x = add_half2(sum_x, in.x);
    sum_y = add_half2(sum_y, in.y);
    sum_z = add_half2(sum_z, in.z);
    sum_w = add_half2(sum_w, in.w);
  }

  // Divide by num_times (time average)
  const float scale = 1.0f / num_times;
  half2x4 accum;
  accum.x = scale_to_half2(sum_x, scale);
  accum.y = scale_to_half2(sum_y, scale);
  accum.z = scale_to_half2(sum_z, scale);
  accum.w = scale_to_half2(sum_w, scale);

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
