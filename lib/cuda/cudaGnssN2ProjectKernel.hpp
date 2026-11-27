/**
 * @file
 * @brief Kernels of cudaGnssN2Project: the lane projection V' = (I - Q Q^H) V (I - Q Q^H) of
 *        selected channels of an n2k correlation frame, in place, in FP32.
 *
 * The frame is cudaCorrelator's: per channel, the 36 lower-triangle 16x16 blocks of 128 inputs,
 * int32 [block][16][16][2], entry (i, j) valid where j <= i (gnssN2ProjectMath.hpp `Layout`).
 * The basis comes from GnssN2Project's basis record (GnssN2Project.hpp): per slot, k orthonormal
 * directions on the stations axis as complex<float>. Placed on the 128-input axis with zeros off
 * the stations (Qt), one projection of the whole matrix gives what GnssN2Project's CPU
 * path writes: the projected station block, the station-other products projected on the station
 * side, and the other inputs' own products untouched.
 *
 * With W = V Qt, M = Qt^H W and Z = W - Qt M / 2, V' = V - Qt Z^H - Z Qt^H. W, M and Z are three
 * strided-batched CGEMMs over the slots (k padded to the largest k of the frame); the update is
 * one kernel that reads and writes the int32 frame in place, rounding as the CPU path does.
 * V is exact in float (an entry is a sum of 8192 4-bit products, at most ~2^20) and Qt is the
 * record's complex<float>; the result is within a count of the CPU path's FP64.
 */
#ifndef CUDA_GNSS_N2_PROJECT_KERNEL_HPP
#define CUDA_GNSS_N2_PROJECT_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace gnss_cuda {

constexpr int N2P_INPUTS = 128;
constexpr int N2P_BS = 16;
constexpr int N2P_BLOCKS = 36;                      ///< lower-triangle 16x16 blocks of 128
constexpr int N2P_WORDS = N2P_BLOCKS * 16 * 16 * 2; ///< int32 per channel of an n2k frame

/// Copy each slot's channel chan[s] of @p frame to out[s] (N2P_WORDS int32 each).
void launch_n2proj_gather(const int32_t* frame, const int* chan, int n_slots, int32_t* out,
                          cudaStream_t stream);

/**
 * The dense Hermitian matrix of each slot's channel, V [slot][128][128] column-major complex
 * float (the diagonal taken as the CPU path's extract() takes it), and the slot's basis on the
 * 128-input axis, Qt [slot][k_ld][128] column-major (column c = direction c; zero off the
 * stations and from the slot's k on). @p rec is the basis record on the device: its stations
 * (@p n of them) and its slots of @p k_slot directions.
 */
void launch_n2proj_dense(const int32_t* frame, const int* chan, const uint8_t* rec, int n,
                         int k_slot, int n_slots, int k_ld, float2* V, float2* Qt,
                         cudaStream_t stream);

/// W = V Qt, M = Qt^H W, Z = W - Qt M / 2 (in W), over the first @p k columns. The handle must be
/// set to the stream the kernels run on. Returns the first failing status, or success.
cublasStatus_t n2proj_wz(cublasHandle_t handle, int n_slots, int k, int k_ld, const float2* V,
                         const float2* Qt, float2* W, float2* M);

/**
 * V' = V - Qt Z^H - Z Qt^H in place on each slot's channel of @p frame, over the lower-triangle
 * entries that involve a station: rounded to nearest (even) and saturated like the CPU path's
 * to_i32, a station's diagonal imaginary part written 0. A slot whose k is 0 is not touched,
 * nor are the other inputs' own products or the upper half of a diagonal block.
 */
void launch_n2proj_update(int32_t* frame, const int* chan, const uint8_t* rec, int n, int k_slot,
                          int n_slots, int k, int k_ld, const float2* Qt, const float2* Z,
                          cudaStream_t stream);

} // namespace gnss_cuda

#endif
