// CUDA kernel calculating the FRB2 beamforming weights W2

#include "cudaCalcFRB2WeightsKernel.hpp"
#include "cudaUtils.hpp"

#include <cstddef>
#include <cuda_fp16.h>

namespace {

// This matches a function defined in Kendrick's beamforming note (eqn 7/8):
//   Ufunc(p, M, theta) = (1/M) sum_{s=0}^{M} A_s cos(pi (2 theta - p) s / M)
// with trapezoid weights A_0 = A_M = 1/2, A_s = 1 otherwise. Instead of evaluating M+1
// cosines like the CPU version in calcFRB2Weights, this uses the cosine recurrence
//   cos((s+1) x) = 2 cos(x) cos(s x) - cos((s-1) x),   x = pi (2 theta - p) / M,
// so only a single cospif call is needed and the loop is two fma per term. The recurrence
// is marginally stable; its error grows at most ~linearly in M (~M eps ~ 1e-5 in float for
// M = 256), far below the float16 resolution (~1e-3) of the output. Results differ from
// the CPU version (cosf(pi * x) summed term by term) at the ~1e-4 level.
//
// A further speedup would expand cos(pi (2 theta - p) s / M) via the angle-addition formula
// into terms depending on (theta, s) and a precomputable P x (M+1) table depending on
// (p, s), turning this function into a dot product (or all of Up into a GEMM). That is only
// a win if the table is tiled through shared memory or handed to cuBLAS (streaming it from
// L2 costs as much as the fma loop), and the recurrence already gets the kernel within a
// few x of the cost of just writing W2, so it is not worth the complexity so far.
__device__ float cuda_frb2_Ufunc(const int p, const int M, const float theta) {
    const float c1 = cospif((2 * theta - p) / M);
    const float two_c1 = 2 * c1;
    float cprev = 1.0f; // cos((s-1) x) for s = 1
    float ccur = c1;    // cos(s x)     for s = 1
    float acc = 0.5f;   // s = 0 term, A_0 = 1/2
    for (int s = 1; s < M; ++s) {
        acc += ccur; // A_s = 1
        const float cnext = fmaf(two_c1, ccur, -cprev);
        cprev = ccur;
        ccur = cnext;
    }
    acc += 0.5f * ccur; // s = M term, A_M = 1/2; after the loop, ccur = cos(M x)
    return acc / M;
}

// One block per (freq, beamR): gridDim.x = num_beams_R, gridDim.y = nfreq. Each block first
// computes Up[P] and Uq[Q] in shared memory, then writes the contiguous P x Q output tile as
// the outer product W2[q][p] = Up[p] * Uq[q].
__global__ void cuda_calc_frb2_weights_kernel(
    const float* __restrict__ const frequencies, const float* __restrict__ const beam_positions,
    float16_t* __restrict__ const W2, const int num_beams_R, const int num_dishes_M,
    const int num_dishes_N, const float sigmaM_x, const float sigmaM_y, const float sigmaM_z,
    const float sigmaN_x, const float sigmaN_y, const float sigmaN_z) {
    const int num_beams_P = 2 * num_dishes_M;
    const int num_beams_Q = 2 * num_dishes_N;

    const int beamR = blockIdx.x;
    const int freq = blockIdx.y;

    const float c0 = 299792458.0f; // speed of light in vacuum [m/s]
    const float wavelength = c0 / frequencies[freq];

    // Unit vector pointing to sky location in the GRID frame
    const float nx = beam_positions[2 * beamR + 0];
    const float ny = beam_positions[2 * beamR + 1];
    const float nz = sqrtf(1 - (nx * nx + ny * ny));

    // Kendrick's FRB beamforming notes, equation 7:
    //   theta = M (nhat . sigma) / lambda
    const float theta_M =
        num_dishes_M * (nx * sigmaM_x + ny * sigmaM_y + nz * sigmaM_z) / wavelength;
    const float theta_N =
        num_dishes_N * (nx * sigmaN_x + ny * sigmaN_y + nz * sigmaN_z) / wavelength;

    // Up and Uq, contiguous in dynamic shared memory
    extern __shared__ float Upq[];
    float* const Up = &Upq[0];
    float* const Uq = &Upq[num_beams_P];

    for (int i = threadIdx.x; i < num_beams_P + num_beams_Q; i += blockDim.x)
        if (i < num_beams_P)
            Up[i] = cuda_frb2_Ufunc(i, num_dishes_M, theta_M);
        else
            Uq[i - num_beams_P] = cuda_frb2_Ufunc(i - num_beams_P, num_dishes_N, theta_N);

    __syncthreads();

    // The output tile for this (freq, beamR) is contiguous, so these writes are coalesced
    const int tile_size = num_beams_P * num_beams_Q;
    float16_t* const tile = W2 + (std::size_t(freq) * num_beams_R + beamR) * std::size_t(tile_size);
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        const int beamP = i % num_beams_P;
        const int beamQ = i / num_beams_P;
        tile[i] = __float2half(Up[beamP] * Uq[beamQ]);
    }
}

} // namespace

void cuda_calc_frb2_weights(const float* const frequencies, const float* const beam_positions,
                            float16_t* const W2, const int nfreq, const int num_beams_R,
                            const int num_dishes_M, const int num_dishes_N, const float sigmaM_x,
                            const float sigmaM_y, const float sigmaM_z, const float sigmaN_x,
                            const float sigmaN_y, const float sigmaN_z, cudaStream_t stream) {
    const int num_beams_P = 2 * num_dishes_M;
    const int num_beams_Q = 2 * num_dishes_N;

    const dim3 nblocks(num_beams_R, nfreq); // gridDim.y must be < 65536
    const int nthreads = 256;
    const std::size_t shmem_bytes = (num_beams_P + num_beams_Q) * sizeof(float);

    cuda_calc_frb2_weights_kernel<<<nblocks, nthreads, shmem_bytes, stream>>>(
        frequencies, beam_positions, W2, num_beams_R, num_dishes_M, num_dishes_N, sigmaM_x,
        sigmaM_y, sigmaM_z, sigmaN_x, sigmaN_y, sigmaN_z);
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}
