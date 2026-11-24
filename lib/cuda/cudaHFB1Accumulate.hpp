// HFB-1 beam accumulator kernel.
//
// Time-averages the FRB1 beamformer output (consumed from the FRB1 stage) into the input of the
// FRB2 stage: it sums `num_times` consecutive time samples per (frequency, beam) and divides by
// `num_times`, collapsing the time axis to a single averaged sample.
//
// See cudaHFB1Accumulate.cu for the kernel implementation.

#ifndef CUDA_HFB1_ACCUMULATE_HPP
#define CUDA_HFB1_ACCUMULATE_HPP

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Free function (not in a namespace) so the `cudaHFB1Accumulate` cudaCommand class can use that
// name; this mirrors the gpu_/cpu_ pairing in cudaQuantizeKernel8.hpp and the launch_ function in
// cudaPLMaskUpchannelizer.hpp.

// Average the FRB1 beams over `num_times` time samples.
//
// Array layouts (`__half`, C index ordering; the beam axes P*Q are the contiguous inner block):
//     input      [num_times][num_frequencies][P*Q]   (read)
//     accumulator           [num_frequencies][P*Q]   (write)
// with the fixed CHIME beam layout P = 2*256, Q = 2*4 (P*Q = 4096). For each (frequency, beam) the
// kernel computes `accumulator = (1/num_times) * sum_t input[t]`.
//
//   accumulator     - output buffer (time-averaged beams)
//   input           - input buffer (FRB1 beams, time-major)
//   num_frequencies - number of frequency channels
//   num_times       - number of input time samples to average
//   stream          - CUDA stream to launch on
void launch_accumulate_hfb1(__half* accumulator, const __half* input, int num_frequencies,
                            int num_times, cudaStream_t stream);

#endif // CUDA_HFB1_ACCUMULATE_HPP
