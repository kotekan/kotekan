// CUDA kernel calculating the FRB2 beamforming weights W2 (see calcFRB2Weights for the CPU
// version and cudaCalcFRB2Weights for the stage wrapping this kernel)

#ifndef CUDA_CALC_FRB2_WEIGHTS_KERNEL_HPP
#define CUDA_CALC_FRB2_WEIGHTS_KERNEL_HPP

#include "DataType.hpp"

#include <cuda_runtime.h>

// Calculate the FRB2 beamforming weights
//   W2[freq][beamR][beamQ][beamP] = Ufunc(beamP, M, theta_M) * Ufunc(beamQ, N, theta_N)
// for `nfreq` frequencies, where P = 2 * num_dishes_M (fastest, contiguous) and
// Q = 2 * num_dishes_N. All array arguments are device pointers:
//   frequencies:    [nfreq] physical frequencies [Hz]
//   beam_positions: [num_beams_R][2] beam direction cosines (nx, ny) in the GRID frame
//   W2:             [nfreq][num_beams_R][2 * num_dishes_N][2 * num_dishes_M]
// sigmaM/sigmaN are the feed separation vectors [m] along the M and N dish grid axes.
// The kernel runs asynchronously on `stream`.
//
// Note: The GPU result is not bit-identical to calcFRB2Weights' CPU result; the GPU uses
// `cospif(x)` where the CPU uses `cosf(pi * x)`, which differ at the ~1e-4 level for the
// large arguments occurring here (the GPU version is the more accurate one).
void cuda_calc_frb2_weights(const float* frequencies, const float* beam_positions, float16_t* W2,
                            int nfreq, int num_beams_R, int num_dishes_M, int num_dishes_N,
                            float sigmaM_x, float sigmaM_y, float sigmaM_z, float sigmaN_x,
                            float sigmaN_y, float sigmaN_z, cudaStream_t stream);

#endif // #ifndef CUDA_CALC_FRB2_WEIGHTS_KERNEL_HPP
