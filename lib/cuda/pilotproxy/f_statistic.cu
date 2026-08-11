/**
 * @file f_statistic.cu
 * @brief GPU-accelerated F-statistic computation for narrowband signal detection.
 *
 * Implements an F-statistic detector built from per-row matched filters
 * (coherent within each K-sample row; row powers are combined incoherently
 * across rows and feeds), optimized for detecting
 * DTV pilot tones in wideband spectral data. Uses fixed-point arithmetic
 * through the dot-product and power-accumulation stages.
 *
 * @section algorithm Algorithm Overview
 *
 * For each detector row and weight term:
 *   1. Compute complex dot product: z[row,term] = sum_k x[m,k] * conj(w[n,k])
 *   2. Accumulate power: P[term] = sum_row |z[row,term]|^2
 *
 * The statistic is a target/reference power ratio:
 *
 *   F = 2 * P[target] / (P[ref1] + P[ref2])
 *
 * Thresholds are supplied by the external calibration/science-tolerance chain
 * and are applied as raw F-statistic thresholds.
 *
 * @section optimization Optimization Strategy
 *
 * - Integer arithmetic in the detector-window loop (avoids float conversion overhead)
 * - Warp shuffle reductions (no shared memory bank conflicts)
 * - Grid-stride loop over detector rows
 * - Weight ROM stores natural matched-filter weights
 * - Packed weights uploaded once per call; optional DP4A path pre-packs lanes
 * - uint64 power accumulation over detector rows (float only at final division/output)
 *
 * @author Dylan
 * @date 2025
 */

#include "config.h"

#define FX_TW_MASTER_NO_HOST_ARRAY
#include "fxfft_master_twiddle.h"
#include "f_statistic.h"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <new>

#define FSTAT_WEIGHT_COUNT (FSTAT_NUM_WEIGHT_TERMS * FSTAT_DETECTOR_WINDOW_SAMPLES)
#define FSTAT_WEIGHT_BYTES (FSTAT_WEIGHT_COUNT * sizeof(InputType))
#define FSTAT_WEIGHT_LANE_COUNT (FSTAT_NUM_WEIGHT_TERMS * FSTAT_DP4A_TAP_PAIRS)

#define FSTAT_CUDA_MAX_THREADS_PER_BLOCK 1024
#define FSTAT_OUTPUT_KERNEL_THREADS 256
#define FSTAT_DP4A_LANE_BITS 8
#define FSTAT_DP4A_LANE1_SHIFT FSTAT_DP4A_LANE_BITS
#define FSTAT_DP4A_LANE2_SHIFT (2 * FSTAT_DP4A_LANE_BITS)
#define FSTAT_DP4A_LANE3_SHIFT (3 * FSTAT_DP4A_LANE_BITS)
#define FSTAT_INT4_MASK ((1 << FSTAT_INT4_COMPONENT_BITS) - 1)
#define FSTAT_INT4_SIGN_BIT (1 << (FSTAT_INT4_COMPONENT_BITS - 1))

static_assert(
    FSTAT_BLOCK_THREADS >= FSTAT_WARP_SIZE,
    "FSTAT_BLOCK_THREADS must be at least one CUDA warp.");
static_assert(
    (FSTAT_BLOCK_THREADS % FSTAT_WARP_SIZE) == 0,
    "FSTAT_BLOCK_THREADS must be a multiple of FSTAT_WARP_SIZE.");
static_assert(
    FSTAT_BLOCK_THREADS <= FSTAT_CUDA_MAX_THREADS_PER_BLOCK,
    "FSTAT_BLOCK_THREADS exceeds the CUDA per-block thread limit.");
static_assert(
    FSTAT_WARPS_PER_BLOCK <= FSTAT_WARP_SIZE,
    "Block reduction assumes at most 32 warps per block.");

#if FSTAT_USE_DP4A
static_assert(
    FSTAT_SAMPLE_BITS_PER_COMPONENT == 4,
    "DP4A path assumes packed complex int4+int4 samples.");
static_assert(
    (FSTAT_DETECTOR_WINDOW_SAMPLES % 2) == 0,
    "DP4A path requires an even detector window.");
#endif

#if FSTAT_USE_DP4A && FSTAT_USE_CONSTANT_WEIGHT_LANES
__constant__ int c_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
#endif

/* ===========================================================================
 * CUDA ERROR CHECKING
 * ===========================================================================*/

static thread_local char g_last_error[512] = "";

static void clear_last_error()
{
    g_last_error[0] = '\0';
}

static void record_api_error(const char* message)
{
    std::snprintf(
        g_last_error,
        sizeof(g_last_error),
        "FStat API error: %s",
        message);
    std::fprintf(stderr, "%s\n", g_last_error);
}

static bool record_cuda_error(cudaError_t err, const char* file, int line)
{
    if (err == cudaSuccess) {
        return true;
    }
    std::snprintf(
        g_last_error,
        sizeof(g_last_error),
        "CUDA error %s:%d: %s",
        file,
        line,
        cudaGetErrorString(err));
    std::fprintf(stderr, "%s\n", g_last_error);
    return false;
}

#define CUDA_CHECK(call) do { \
    if (!record_cuda_error((call), __FILE__, __LINE__)) { \
        return; \
    } \
} while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define CUDA_CHECK_BOOL(call) do { \
    if (!record_cuda_error((call), __FILE__, __LINE__)) { \
        return false; \
    } \
} while (0)

#define CUDA_CHECK_LAST_BOOL() CUDA_CHECK_BOOL(cudaGetLastError())

#ifndef NDEBUG
#define CUDA_CHECK_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#define CUDA_CHECK_SYNC_BOOL() CUDA_CHECK_BOOL(cudaDeviceSynchronize())
#else
#define CUDA_CHECK_SYNC() do { } while (0)
#define CUDA_CHECK_SYNC_BOOL() do { } while (0)
#endif

/* ===========================================================================
 * DEVICE HELPER FUNCTIONS
 * ===========================================================================*/

/**
 * @brief Sign-extend a packed n-bit two's-complement component.
 */
__device__ __forceinline__
int sign_extend_nbits(int x, int bits)
{
    const int mask = (1 << bits) - 1;
    const int sign = 1 << (bits - 1);

    x &= mask;
    return (x ^ sign) - sign;
}

/**
 * @brief Unpack a packed complex sample to real/imag components.
 *
 * Handles sign extension for both components:
 *   - Real: Upper FSTAT_SAMPLE_BITS_PER_COMPONENT bits
 *   - Imag: Lower FSTAT_SAMPLE_BITS_PER_COMPONENT bits
 *
 * @param packed  Packed complex value
 * @return short2 with .x = real, .y = imag
 */
__device__ __forceinline__
short2 unpack_sample(InputType packed)
{
    constexpr int bits = FSTAT_SAMPLE_BITS_PER_COMPONENT;
    constexpr int mask = (1 << bits) - 1;

    const int byte = static_cast<int>(static_cast<unsigned char>(packed));

    const int real = sign_extend_nbits(byte >> bits, bits);
    const int imag = sign_extend_nbits(byte & mask, bits);

    return make_short2(
        static_cast<short>(real),
        static_cast<short>(imag));
}

/**
 * @brief Complex multiply by the conjugate of a matched-filter weight.
 *
 * Computes x times conj(w) using integer arithmetic.
 * Result uses 32-bit integers to hold 16x16 products without overflow.
 *
 * @param x  Input sample (real, imag)
 * @param w  Natural matched-filter weight
 * @return int2 with .x = real part, .y = imag part
 */
__device__ __forceinline__
int2 complex_multiply(short2 x, short2 w)
{
    // (x.r + j*x.i) x conj(w.r + j*w.i)
    // = (x.r*w.r + x.i*w.i) + j*(x.i*w.r - x.r*w.i)
    return make_int2(
        static_cast<int>(x.x) * w.x + static_cast<int>(x.y) * w.y,  // Real
        static_cast<int>(x.y) * w.x - static_cast<int>(x.x) * w.y   // Imag
    );
}

/**
 * @brief Complex addition (integer).
 */
__device__ __forceinline__
int2 complex_add(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

#if FSTAT_USE_DP4A
/**
 * @brief Sign-extend a packed 4-bit two's-complement component.
 */
__device__ __forceinline__
int sign_extend_i4(int x)
{
    return sign_extend_nbits(x, FSTAT_INT4_COMPONENT_BITS);
}

/**
 * @brief Pack four signed int8 lanes into the 32-bit format consumed by DP4A.
 */
__device__ __forceinline__
int pack4_i8(int a0, int a1, int a2, int a3)
{
    unsigned int u0 = static_cast<unsigned char>(static_cast<signed char>(a0));
    unsigned int u1 = static_cast<unsigned char>(static_cast<signed char>(a1));
    unsigned int u2 = static_cast<unsigned char>(static_cast<signed char>(a2));
    unsigned int u3 = static_cast<unsigned char>(static_cast<signed char>(a3));
    return static_cast<int>(
        u0
        | (u1 << FSTAT_DP4A_LANE1_SHIFT)
        | (u2 << FSTAT_DP4A_LANE2_SHIFT)
        | (u3 << FSTAT_DP4A_LANE3_SHIFT));
}

/**
 * @brief Decode two packed complex samples into DP4A lanes.
 *
 * For x * conj(w):
 *   real = [xr0, xi0, xr1, xi1] dot [wr0, wi0, wr1, wi1]
 *   imag = [xi0,-xr0, xi1,-xr1] dot [wr0, wi0, wr1, wi1]
 *
 * The DP4A lanes are int8 containers. They contain signed int4 values
 * and, for the imaginary lane, negated signed int4 values. If a data
 * component is -8, its negation is +8, which is representable in int8.
 * The numerical product bound is still governed by the int4 container
 * magnitude, so the 4 + 4 + 1 + log2(128) = 16 bit-growth argument holds.
 */
__device__ __forceinline__
void unpack_two_complex_bytes_to_dp4a_lanes(
    InputType x0,
    InputType x1,
    int& a_re,
    int& a_im)
{
    int b0 = static_cast<int>(static_cast<unsigned char>(x0));
    int b1 = static_cast<int>(static_cast<unsigned char>(x1));

    int x0r = sign_extend_i4(b0 >> FSTAT_INT4_COMPONENT_BITS);
    int x0i = sign_extend_i4(b0);
    int x1r = sign_extend_i4(b1 >> FSTAT_INT4_COMPONENT_BITS);
    int x1i = sign_extend_i4(b1);

    a_re = pack4_i8(x0r, x0i, x1r, x1i);
    a_im = pack4_i8(x0i, -x0r, x1i, -x1r);
}

/**
 * @brief Load a prepacked DP4A weight lane.
 *
 * The DP4A path can load the tiny 768-byte lane table from CUDA constant
 * memory or from a per-handle global buffer preloaded into shared memory. The
 * default production configuration uses shared-memory preload because
 * different threads in a warp read different pair indices.
 */
__device__ __forceinline__
int load_weight_lane(const int* W_Lanes, int idx)
{
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    return c_weight_lanes[idx];
    #else
    return W_Lanes[idx];
    #endif
}
#endif

/**
 * @brief Warp-level sum reduction for int2.
 *
 * Uses shuffle intrinsics for fast intra-warp communication.
 * Result is valid only in lane 0 after completion.
 *
 * @param val  Per-thread value to reduce
 * @return Warp sum (valid in lane 0 only)
 */
__device__ __forceinline__
int2 warp_reduce_sum(int2 val)
{
    #pragma unroll
    for (int offset = FSTAT_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(FSTAT_WARP_MASK, val.x, offset);
        val.y += __shfl_down_sync(FSTAT_WARP_MASK, val.y, offset);
    }
    return val;
}

#ifndef NDEBUG
__device__ __forceinline__
void debug_check_dot_int16_bounds(
    const int2* dot,
    int detector_row,
    int batch_index)
{
    #pragma unroll
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        if (dot[n].x < SHRT_MIN || dot[n].x > SHRT_MAX ||
            dot[n].y < SHRT_MIN || dot[n].y > SHRT_MAX) {
            if (batch_index >= 0) {
                printf(
                    "FStat dot product exceeded int16 bound: batch=%d row=%d term=%d real=%d imag=%d\n",
                    batch_index,
                    detector_row,
                    n,
                    dot[n].x,
                    dot[n].y);
            } else {
                printf(
                    "FStat dot product exceeded int16 bound: row=%d term=%d real=%d imag=%d\n",
                    detector_row,
                    n,
                    dot[n].x,
                    dot[n].y);
            }
        }
    }
}
#endif

/* ===========================================================================
 * CUDA KERNELS
 * ===========================================================================*/

/**
 * @brief Main power accumulation kernel.
 *
 * Computes power in each weight vector's matched filter output,
 * accumulated over all detector rows in one block.
 *
 * Algorithm per detector block:
 *   1. Grid-stride loop over detector rows
 *   2. Thread-stride loop over detector-window samples (complex dot product)
 *   3. Warp reduction -> shared memory -> block reduction
 *   4. Accumulate |dot_product|^2 into per-block uint64 accumulator
 *   5. Atomic add block results to global integer power scratch
 *
 * @param X            Row-major input samples [detector_rows_per_block x detector_window_samples],
 *                     packed complex, with X[m, k] = X[m * detector_window_samples + k]
 * @param Power_Terms  Output power per weight vector [num_weight_terms]
 * @param detector_rows_per_block  Number of detector rows in the matrix view
 */
__global__ void
kernel_accumulate_power(
    const InputType* __restrict__ X,
    const InputType* __restrict__ W,
    const int*       __restrict__ W_Lanes,
    unsigned long long* __restrict__ Power_Terms,
    int detector_rows_per_block)
{
    #if FSTAT_USE_DP4A
    (void)W;
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    #endif
    #else
    (void)W_Lanes;
    #endif

    // Shared memory for inter-warp reduction
    __shared__ int2 warp_sums[FSTAT_NUM_WEIGHT_TERMS][FSTAT_WARPS_PER_BLOCK];
    __shared__ unsigned long long block_power[FSTAT_NUM_WEIGHT_TERMS];

    const int tid     = threadIdx.x;
    const int warp_id = tid / FSTAT_WARP_SIZE;
    const int lane_id = tid % FSTAT_WARP_SIZE;

    if (tid < FSTAT_NUM_WEIGHT_TERMS) {
        block_power[tid] = 0ULL;
    }
    __syncthreads();

    #if FSTAT_USE_DP4A && FSTAT_USE_SHARED_WEIGHT_LANES && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    const int* weight_lanes = W_Lanes;
    __shared__ int shared_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
    for (int idx = tid; idx < FSTAT_WEIGHT_LANE_COUNT; idx += blockDim.x) {
        shared_weight_lanes[idx] = W_Lanes[idx];
    }
    __syncthreads();
    weight_lanes = shared_weight_lanes;
    #elif FSTAT_USE_DP4A
    const int* weight_lanes = W_Lanes;
    #endif

    // -------------------------------------------------------------------------
    // Grid-stride loop over detector rows
    // -------------------------------------------------------------------------
    for (int m = blockIdx.x; m < detector_rows_per_block; m += gridDim.x) {

        // Per-thread dot product accumulators
        int2 dot[FSTAT_NUM_WEIGHT_TERMS];
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = make_int2(0, 0);
        }

        // ---------------------------------------------------------------------
        // Thread-stride loop over frequency taps
        // ---------------------------------------------------------------------
        #if FSTAT_USE_DP4A
        for (int pair = tid; pair < FSTAT_DP4A_TAP_PAIRS; pair += blockDim.x) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int a_re;
            int a_im;
            unpack_two_complex_bytes_to_dp4a_lanes(
                X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0],
                X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k1],
                a_re,
                a_im);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                const int w_lane =
                    load_weight_lane(weight_lanes, n * FSTAT_DP4A_TAP_PAIRS + pair);
                dot[n].x = __dp4a(a_re, w_lane, dot[n].x);
                dot[n].y = __dp4a(a_im, w_lane, dot[n].y);
            }
        }
        #else
        for (int k = tid; k < FSTAT_DETECTOR_WINDOW_SAMPLES; k += blockDim.x) {
            short2 x_val = unpack_sample(X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                short2 w_val = unpack_sample(W[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
                dot[n] = complex_add(dot[n], complex_multiply(x_val, w_val));
            }
        }
        #endif

        // ---------------------------------------------------------------------
        // Warp reduction
        // ---------------------------------------------------------------------
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = warp_reduce_sum(dot[n]);
        }

        // Write warp results to shared memory
        if (lane_id == 0) {
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                warp_sums[n][warp_id] = dot[n];
            }
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Block reduction (warp 0 only)
        // ---------------------------------------------------------------------
        if (warp_id == 0) {
            // Load from shared memory
            if (lane_id < FSTAT_WARPS_PER_BLOCK) {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = warp_sums[n][lane_id];
                }
            } else {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = make_int2(0, 0);
                }
            }

            // Final warp reduction
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                dot[n] = warp_reduce_sum(dot[n]);
            }

            // Thread 0: compute magnitude squared and accumulate
            if (tid == 0) {
                #ifndef NDEBUG
                debug_check_dot_int16_bounds(dot, m, -1);
                #endif

                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    // Use 64-bit to prevent overflow in |z|^2 computation
                    long long zr = static_cast<long long>(dot[n].x);
                    long long zi = static_cast<long long>(dot[n].y);
                    unsigned long long mag_sq =
                        static_cast<unsigned long long>(zr * zr + zi * zi);

                    block_power[n] += mag_sq;
                }
            }
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Write block results to global memory
    // -------------------------------------------------------------------------
    if (tid == 0) {
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            atomicAdd(&Power_Terms[n], block_power[n]);
        }
    }
}

/**
 * @brief Batched power accumulation kernel.
 *
 * Processes `batch` independent row-major input blocks, each of shape
 * [detector_rows_per_block x detector_window_samples].
 * The batch index is mapped to blockIdx.y.
 *
 * @param X            Row-major input samples [batch x detector_rows_per_block x detector_window_samples],
 *                     packed complex
 * @param Power_Terms  Output power per weight vector [batch x num_weight_terms]
 * @param detector_rows_per_block  Number of detector rows in the matrix view per block
 * @param batch        Number of independent blocks
 */
__global__ void
kernel_accumulate_power_batched(
    const InputType* __restrict__ X,
    const InputType* __restrict__ W,
    const int*       __restrict__ W_Lanes,
    unsigned long long* __restrict__ Power_Terms,
    int detector_rows_per_block,
    int batch)
{
    #if FSTAT_USE_DP4A
    (void)W;
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    #endif
    #else
    (void)W_Lanes;
    #endif

    const int b = blockIdx.y;
    if (b >= batch) {
        return;
    }

    const size_t batch_stride = static_cast<size_t>(detector_rows_per_block) * FSTAT_DETECTOR_WINDOW_SAMPLES;
    const InputType* Xb = X + batch_stride * static_cast<size_t>(b);
    unsigned long long* P = Power_Terms + (b * FSTAT_NUM_WEIGHT_TERMS);

    // Shared memory for inter-warp reduction
    __shared__ int2 warp_sums[FSTAT_NUM_WEIGHT_TERMS][FSTAT_WARPS_PER_BLOCK];
    __shared__ unsigned long long block_power[FSTAT_NUM_WEIGHT_TERMS];

    const int tid     = threadIdx.x;
    const int warp_id = tid / FSTAT_WARP_SIZE;
    const int lane_id = tid % FSTAT_WARP_SIZE;

    if (tid < FSTAT_NUM_WEIGHT_TERMS) {
        block_power[tid] = 0ULL;
    }
    __syncthreads();

    #if FSTAT_USE_DP4A && FSTAT_USE_SHARED_WEIGHT_LANES && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    const int* weight_lanes = W_Lanes;
    __shared__ int shared_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
    for (int idx = tid; idx < FSTAT_WEIGHT_LANE_COUNT; idx += blockDim.x) {
        shared_weight_lanes[idx] = W_Lanes[idx];
    }
    __syncthreads();
    weight_lanes = shared_weight_lanes;
    #elif FSTAT_USE_DP4A
    const int* weight_lanes = W_Lanes;
    #endif

    for (int m = blockIdx.x; m < detector_rows_per_block; m += gridDim.x) {
        int2 dot[FSTAT_NUM_WEIGHT_TERMS];
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = make_int2(0, 0);
        }

        #if FSTAT_USE_DP4A
        for (int pair = tid; pair < FSTAT_DP4A_TAP_PAIRS; pair += blockDim.x) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int a_re;
            int a_im;
            unpack_two_complex_bytes_to_dp4a_lanes(
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0],
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k1],
                a_re,
                a_im);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                const int w_lane =
                    load_weight_lane(weight_lanes, n * FSTAT_DP4A_TAP_PAIRS + pair);
                dot[n].x = __dp4a(a_re, w_lane, dot[n].x);
                dot[n].y = __dp4a(a_im, w_lane, dot[n].y);
            }
        }
        #else
        for (int k = tid; k < FSTAT_DETECTOR_WINDOW_SAMPLES; k += blockDim.x) {
            short2 x_val = unpack_sample(Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                short2 w_val = unpack_sample(W[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
                dot[n] = complex_add(dot[n], complex_multiply(x_val, w_val));
            }
        }
        #endif

        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = warp_reduce_sum(dot[n]);
        }

        if (lane_id == 0) {
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                warp_sums[n][warp_id] = dot[n];
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < FSTAT_WARPS_PER_BLOCK) {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = warp_sums[n][lane_id];
                }
            } else {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = make_int2(0, 0);
                }
            }

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                dot[n] = warp_reduce_sum(dot[n]);
            }

            if (tid == 0) {
                #ifndef NDEBUG
                debug_check_dot_int16_bounds(dot, m, b);
                #endif

                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    long long zr = static_cast<long long>(dot[n].x);
                    long long zi = static_cast<long long>(dot[n].y);
                    unsigned long long mag_sq =
                        static_cast<unsigned long long>(zr * zr + zi * zi);
                    block_power[n] += mag_sq;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            atomicAdd(&P[n], block_power[n]);
        }
    }
}

/**
 * @brief Emit exact complex per-row dot products (v2 time-coherent front end).
 *
 * Identical dot-product and reduction structure to kernel_accumulate_power,
 * but instead of squaring and accumulating |z|^2, thread 0 of the reducing
 * warp writes the exact int32 complex row sum for every weight term:
 *
 *   RowSums[(n * detector_rows_per_block + m) * 2 + 0] = Re z[n, m]
 *   RowSums[(n * detector_rows_per_block + m) * 2 + 1] = Im z[n, m]
 *
 * Layout contract (stage-2 consumers depend on this):
 *   - term-major: each weight term's rows are contiguous;
 *   - rows keep the packer's stream-major order (each input stream's
 *     windows are contiguous), so a term slice reshapes to
 *     [num_streams, windows_per_stream] with no gather;
 *   - interleaved re/im int32 pairs.
 *
 * Values are exact integers (|component| <= detector_window_samples *
 * 2 * 7 * 8 = 14336 for the locked 4-bit path), so output is
 * bit-reproducible regardless of accumulation order. The all-bin marginal
 * sum over |z|^2 reproduces the v1 power terms exactly in int64.
 */
__global__ void
kernel_accumulate_row_sums(
    const InputType* __restrict__ X,
    const InputType* __restrict__ W,
    const int*       __restrict__ W_Lanes,
    int*             __restrict__ RowSums,
    int detector_rows_per_block)
{
    #if FSTAT_USE_DP4A
    (void)W;
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    #endif
    #else
    (void)W_Lanes;
    #endif

    __shared__ int2 warp_sums[FSTAT_NUM_WEIGHT_TERMS][FSTAT_WARPS_PER_BLOCK];

    const int tid     = threadIdx.x;
    const int warp_id = tid / FSTAT_WARP_SIZE;
    const int lane_id = tid % FSTAT_WARP_SIZE;

    #if FSTAT_USE_DP4A && FSTAT_USE_SHARED_WEIGHT_LANES && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    const int* weight_lanes = W_Lanes;
    __shared__ int shared_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
    for (int idx = tid; idx < FSTAT_WEIGHT_LANE_COUNT; idx += blockDim.x) {
        shared_weight_lanes[idx] = W_Lanes[idx];
    }
    __syncthreads();
    weight_lanes = shared_weight_lanes;
    #elif FSTAT_USE_DP4A
    const int* weight_lanes = W_Lanes;
    #endif

    for (int m = blockIdx.x; m < detector_rows_per_block; m += gridDim.x) {

        int2 dot[FSTAT_NUM_WEIGHT_TERMS];
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = make_int2(0, 0);
        }

        #if FSTAT_USE_DP4A
        for (int pair = tid; pair < FSTAT_DP4A_TAP_PAIRS; pair += blockDim.x) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int a_re;
            int a_im;
            unpack_two_complex_bytes_to_dp4a_lanes(
                X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0],
                X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k1],
                a_re,
                a_im);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                const int w_lane =
                    load_weight_lane(weight_lanes, n * FSTAT_DP4A_TAP_PAIRS + pair);
                dot[n].x = __dp4a(a_re, w_lane, dot[n].x);
                dot[n].y = __dp4a(a_im, w_lane, dot[n].y);
            }
        }
        #else
        for (int k = tid; k < FSTAT_DETECTOR_WINDOW_SAMPLES; k += blockDim.x) {
            short2 x_val = unpack_sample(X[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                short2 w_val = unpack_sample(W[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
                dot[n] = complex_add(dot[n], complex_multiply(x_val, w_val));
            }
        }
        #endif

        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = warp_reduce_sum(dot[n]);
        }

        if (lane_id == 0) {
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                warp_sums[n][warp_id] = dot[n];
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < FSTAT_WARPS_PER_BLOCK) {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = warp_sums[n][lane_id];
                }
            } else {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = make_int2(0, 0);
                }
            }

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                dot[n] = warp_reduce_sum(dot[n]);
            }

            if (tid == 0) {
                #ifndef NDEBUG
                debug_check_dot_int16_bounds(dot, m, -1);
                #endif

                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    const size_t out_index =
                        (static_cast<size_t>(n) * detector_rows_per_block + m)
                        * 2;
                    RowSums[out_index + 0] = dot[n].x;
                    RowSums[out_index + 1] = dot[n].y;
                }
            }
        }
        __syncthreads();
    }
}

/**
 * @brief Batched exact complex row-sum emission.
 *
 * Batch entry b (blockIdx.y) reads input block b and writes
 * RowSums + b * num_weight_terms * detector_rows_per_block * 2, preserving
 * the per-block term-major/stream-major/interleaved layout contract of
 * kernel_accumulate_row_sums.
 */
__global__ void
kernel_accumulate_row_sums_batched(
    const InputType* __restrict__ X,
    const InputType* __restrict__ W,
    const int*       __restrict__ W_Lanes,
    int*             __restrict__ RowSums,
    int detector_rows_per_block,
    int batch)
{
    #if FSTAT_USE_DP4A
    (void)W;
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    #endif
    #else
    (void)W_Lanes;
    #endif

    const int b = blockIdx.y;
    if (b >= batch) {
        return;
    }

    const size_t batch_stride = static_cast<size_t>(detector_rows_per_block) * FSTAT_DETECTOR_WINDOW_SAMPLES;
    const InputType* Xb = X + batch_stride * static_cast<size_t>(b);
    int* Out = RowSums
        + static_cast<size_t>(b) * FSTAT_NUM_WEIGHT_TERMS
              * static_cast<size_t>(detector_rows_per_block) * 2;

    __shared__ int2 warp_sums[FSTAT_NUM_WEIGHT_TERMS][FSTAT_WARPS_PER_BLOCK];

    const int tid     = threadIdx.x;
    const int warp_id = tid / FSTAT_WARP_SIZE;
    const int lane_id = tid % FSTAT_WARP_SIZE;

    #if FSTAT_USE_DP4A && FSTAT_USE_SHARED_WEIGHT_LANES && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    const int* weight_lanes = W_Lanes;
    __shared__ int shared_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
    for (int idx = tid; idx < FSTAT_WEIGHT_LANE_COUNT; idx += blockDim.x) {
        shared_weight_lanes[idx] = W_Lanes[idx];
    }
    __syncthreads();
    weight_lanes = shared_weight_lanes;
    #elif FSTAT_USE_DP4A
    const int* weight_lanes = W_Lanes;
    #endif

    for (int m = blockIdx.x; m < detector_rows_per_block; m += gridDim.x) {

        int2 dot[FSTAT_NUM_WEIGHT_TERMS];
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = make_int2(0, 0);
        }

        #if FSTAT_USE_DP4A
        for (int pair = tid; pair < FSTAT_DP4A_TAP_PAIRS; pair += blockDim.x) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int a_re;
            int a_im;
            unpack_two_complex_bytes_to_dp4a_lanes(
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0],
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k1],
                a_re,
                a_im);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                const int w_lane =
                    load_weight_lane(weight_lanes, n * FSTAT_DP4A_TAP_PAIRS + pair);
                dot[n].x = __dp4a(a_re, w_lane, dot[n].x);
                dot[n].y = __dp4a(a_im, w_lane, dot[n].y);
            }
        }
        #else
        for (int k = tid; k < FSTAT_DETECTOR_WINDOW_SAMPLES; k += blockDim.x) {
            short2 x_val = unpack_sample(Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                short2 w_val = unpack_sample(W[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
                dot[n] = complex_add(dot[n], complex_multiply(x_val, w_val));
            }
        }
        #endif

        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = warp_reduce_sum(dot[n]);
        }

        if (lane_id == 0) {
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                warp_sums[n][warp_id] = dot[n];
            }
        }
        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < FSTAT_WARPS_PER_BLOCK) {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = warp_sums[n][lane_id];
                }
            } else {
                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    dot[n] = make_int2(0, 0);
                }
            }

            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                dot[n] = warp_reduce_sum(dot[n]);
            }

            if (tid == 0) {
                #ifndef NDEBUG
                debug_check_dot_int16_bounds(dot, m, b);
                #endif

                #pragma unroll
                for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                    const size_t out_index =
                        (static_cast<size_t>(n) * detector_rows_per_block + m)
                        * 2;
                    Out[out_index + 0] = dot[n].x;
                    Out[out_index + 1] = dot[n].y;
                }
            }
        }
        __syncthreads();
    }
}

/**
 * @brief Compute F-statistic from accumulated powers.
 *
 * F = 2 x P[target] / (P[ref1] + P[ref2])
 *
 * @param P         Power terms [num_weight_terms] from accumulation kernel
 * @param F_Result  Output F-statistic (single value)
 */
__global__ void
kernel_compute_f_statistic(
    const unsigned long long* __restrict__ P,
    float*       __restrict__ F_Result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double denominator =
            static_cast<double>(P[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX])
            + static_cast<double>(P[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX]);

        if (denominator > 0.0) {
            double numerator =
                FSTAT_RAW_NUMDEN_SCALE
                * static_cast<double>(P[FSTAT_TARGET_WEIGHT_INDEX]);
            *F_Result = static_cast<float>(numerator / denominator);
        } else {
            *F_Result = 0.0f;  // Degenerate case: no reference power
        }
    }
}

/**
 * @brief Compute F-statistic for each batch entry.
 *
 * @param P         Power terms [batch x num_weight_terms] from accumulation kernel
 * @param F_Result  Output F-statistics [batch]
 * @param batch     Number of batch entries
 */
__global__ void
kernel_compute_f_statistic_batched(
    const unsigned long long* __restrict__ P,
    float*       __restrict__ F_Result,
    int batch)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }
    const unsigned long long* Pb = P + (b * FSTAT_NUM_WEIGHT_TERMS);
    double denominator =
        static_cast<double>(Pb[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX])
        + static_cast<double>(Pb[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX]);
    if (denominator > 0.0) {
        double numerator =
            FSTAT_RAW_NUMDEN_SCALE
            * static_cast<double>(Pb[FSTAT_TARGET_WEIGHT_INDEX]);
        F_Result[b] = static_cast<float>(numerator / denominator);
    } else {
        F_Result[b] = 0.0f;
    }
}

/**
 * @brief Convert integer power terms to a diagnostic float output buffer.
 */
__global__ void
kernel_convert_power_terms_to_float(
    const unsigned long long* __restrict__ P,
    float* __restrict__ Out,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    Out[idx] = static_cast<float>(P[idx]);
}

/**
 * @brief Saturating uint64 multiply for defensive rational-threshold products.
 *
 * The locked detector's shelf thresholds and power ranges are expected to fit
 * without saturation. Saturation is defensive only; any debug-build saturation
 * report should be treated as a validation failure for deployed thresholds.
 */
__device__ __forceinline__
void record_rational_overflow(unsigned int* overflow_count)
{
    if (overflow_count != nullptr) {
        atomicAdd(overflow_count, 1u);
    }
}

__device__ __forceinline__
unsigned long long saturating_mul_u64(
    unsigned long long a,
    unsigned long long b,
    unsigned int* overflow_count)
{
    constexpr unsigned long long max_u64 = ~0ULL;
    if (a != 0ULL && b > max_u64 / a) {
        record_rational_overflow(overflow_count);
        #ifndef NDEBUG
        printf(
            "FStat rational threshold multiply saturated: %llu * %llu\n",
            a,
            b);
        #endif
        return max_u64;
    }
    return a * b;
}

/**
 * @brief Saturating uint64 add for defensive reference-power sums.
 */
__device__ __forceinline__
unsigned long long saturating_add_u64(
    unsigned long long a,
    unsigned long long b,
    unsigned int* overflow_count)
{
    constexpr unsigned long long max_u64 = ~0ULL;
    if (b > max_u64 - a) {
        record_rational_overflow(overflow_count);
        #ifndef NDEBUG
        printf(
            "FStat rational threshold add saturated: %llu + %llu\n",
            a,
            b);
        #endif
        return max_u64;
    }
    return a + b;
}

/**
 * @brief Write P_target, P_ref1 + P_ref2, and half-threshold mask decisions.
 *
 * Applies, when P_ref1 + P_ref2 is nonzero:
 *
 *     P_target / (P_ref1 + P_ref2) >=
 *         threshold_half_num / threshold_half_den
 *
 * where threshold_half_num / threshold_half_den is one half of the full raw
 * F-statistic threshold. If P_ref1 + P_ref2 is zero, the reference power is
 * invalid for the deployed detector and the mask is forced to zero.
 */
__global__ void
kernel_write_num_den_mask_threshold_half_rational(
    const unsigned long long* __restrict__ P,
    unsigned long long* __restrict__ Num_Result,
    unsigned long long* __restrict__ Den_Result,
    unsigned char* __restrict__ Mask_Result,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    int batch,
    unsigned int* __restrict__ rational_overflow_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }

    const unsigned long long* Pb = P + (idx * FSTAT_NUM_WEIGHT_TERMS);
    const unsigned long long denominator = saturating_add_u64(
        Pb[FSTAT_LOWER_REFERENCE_WEIGHT_INDEX],
        Pb[FSTAT_UPPER_REFERENCE_WEIGHT_INDEX],
        rational_overflow_count);

    Num_Result[idx] = Pb[FSTAT_TARGET_WEIGHT_INDEX];
    Den_Result[idx] = denominator;

    if (threshold_half_den == 0ULL || denominator == 0ULL) {
        Mask_Result[idx] = 0u;
        return;
    }

    const unsigned long long lhs = saturating_mul_u64(
        Pb[FSTAT_TARGET_WEIGHT_INDEX],
        threshold_half_den,
        rational_overflow_count);
    const unsigned long long rhs = saturating_mul_u64(
        threshold_half_num,
        denominator,
        rational_overflow_count);
    Mask_Result[idx] = (lhs > rhs) ? 1u : 0u;  /* strict: contract says positive excess */
}


/* ===========================================================================
 * FXFFT256 V1 FINE-REDUCTION STAGE (kernel core 2.1.0)
 * ===========================================================================
 * Frozen deterministic fixed-point FFT over the v2 row-sum buffer, plus
 * exact uint64 feed-power sums. The specification, Q15 twiddle table, and
 * rounding rule are frozen as fxfft256 v1: reference
 * src/pilot_proxy/fxfft.py, port template cuda/fxfft256_ref.c, golden
 * vectors tests/data/fxfft256_golden_v1.npz. This device implementation
 * must reproduce those outputs bit-for-bit
 * (tests/kernel/test_fine_powers_gpu.py); any arithmetic change here is a
 * spec change and must be versioned, not absorbed.
 */

#define FSTAT_FXFFT_SHIFT 15
#define FSTAT_FXFFT_ROUND (1LL << 14)

/* Q15 twiddles, shared with the Python reference and the C reference via the
 * generated master table (tools/emit_fxfft_tables.py). Literal constants,
 * never runtime trigonometry. A length-n radix-2 DIT reads
 * master[t << (FX_MASTER_LOG2 - stage)]; the decimation stride folds into the
 * shift, so at FSTAT_FINE_NUM_BINS = 256 the entries selected are exactly the
 * ones the frozen 128-entry fxfft256 v1 table held -- identical integers, not
 * a re-derivation. Constant-bank cost is FX_MASTER_HALF * 8 = 8 KiB. */
__device__ __constant__ int fstat_fxfft_twiddle_q15[FX_MASTER_HALF][2] =
    FX_TW_MASTER_INIT;

__device__ __forceinline__ int fstat_fxfft_round15(long long v)
{
    /* floor((v + 2^14) / 2^15) via arithmetic shift (nvcc semantics). */
    return static_cast<int>((v + FSTAT_FXFFT_ROUND) >> FSTAT_FXFFT_SHIFT);
}

__device__ __forceinline__ unsigned fstat_fxfft_bitrev(unsigned i)
{
    unsigned r = 0;
    for (unsigned k = 0; k < (unsigned)FSTAT_FINE_LOG2; ++k) {
        r |= ((i >> k) & 1u) << ((unsigned)FSTAT_FINE_LOG2 - 1u - k);
    }
    return r;
}

/**
 * @brief One stream's frozen transform: 128 interleaved (re,im) int32
 *        window sums -> natural-order x[256][2] spectrum.
 *
 * Bit-identical to fxfft256 in src/pilot_proxy/fxfft.py. No-overflow
 * contract: |input components| <= 2^20 keeps every working value below
 * 2^31 (kernel row sums are bounded at 14336).
 */
__device__ void fstat_fxfft256_device(
    const int* __restrict__ src,
    int x[FSTAT_FINE_NUM_BINS][2])
{
    for (unsigned i = 0; i < FSTAT_FINE_NUM_BINS; ++i) {
        const unsigned s = fstat_fxfft_bitrev(i);
        if (s < FSTAT_FINE_WINDOWS_PER_STREAM) {
            x[i][0] = src[2u * s];
            x[i][1] = src[2u * s + 1u];
        } else {
            x[i][0] = 0;
            x[i][1] = 0;
        }
    }
    for (unsigned stage = 1; stage <= (unsigned)FSTAT_FINE_LOG2; ++stage) {
        const unsigned m = 1u << stage;
        const unsigned half = m >> 1;
        for (unsigned j0 = 0; j0 < FSTAT_FINE_NUM_BINS; j0 += m) {
            for (unsigned t = 0; t < half; ++t) {
                const int* w = fstat_fxfft_twiddle_q15[t << ((unsigned)FX_MASTER_LOG2 - stage)];
                const long long c = w[0];
                const long long sn = w[1];
                const long long br = x[j0 + t + half][0];
                const long long bi = x[j0 + t + half][1];
                const int tr = fstat_fxfft_round15(br * c - bi * sn);
                const int ti = fstat_fxfft_round15(bi * c + br * sn);
                const int ar = x[j0 + t][0];
                const int ai = x[j0 + t][1];
                x[j0 + t][0] = ar + tr;
                x[j0 + t][1] = ai + ti;
                x[j0 + t + half][0] = ar - tr;
                x[j0 + t + half][1] = ai - ti;
            }
        }
    }
}

/**
 * @brief Fine-power accumulation over the row-sum buffer.
 *
 * Grid: (stream chunks, num_weight_terms, batch); block: FSTAT_BLOCK_THREADS.
 * Each thread transforms one stream and accumulates |X[b]|^2 into a shared
 * per-block uint64 spectrum; blocks flush with global 64-bit atomics.
 * Every addition is an exact integer, so the accumulated result is
 * bit-reproducible for any launch schedule or block decomposition.
 */
__global__ void kernel_fine_powers(
    const int* __restrict__ row_sums,
    unsigned long long* __restrict__ fine_out,
    int num_streams,
    int batch)
{
    __shared__ unsigned long long acc[FSTAT_FINE_NUM_BINS];
    const int term = blockIdx.y;
    const int b = blockIdx.z;

    for (int i = threadIdx.x; i < FSTAT_FINE_NUM_BINS; i += blockDim.x) {
        acc[i] = 0ULL;
    }
    __syncthreads();

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_streams) {
        const size_t rows =
            static_cast<size_t>(num_streams) * FSTAT_FINE_WINDOWS_PER_STREAM;
        const int* src = row_sums
            + ((static_cast<size_t>(b) * FSTAT_NUM_WEIGHT_TERMS + term) * rows
               + static_cast<size_t>(s) * FSTAT_FINE_WINDOWS_PER_STREAM) * 2u;
        int x[FSTAT_FINE_NUM_BINS][2];
        fstat_fxfft256_device(src, x);
        for (int bin = 0; bin < FSTAT_FINE_NUM_BINS; ++bin) {
            const long long re = x[bin][0];
            const long long im = x[bin][1];
            atomicAdd(&acc[bin],
                      static_cast<unsigned long long>(re * re + im * im));
        }
    }
    __syncthreads();

    unsigned long long* out = fine_out
        + (static_cast<size_t>(b) * FSTAT_NUM_WEIGHT_TERMS + term)
          * FSTAT_FINE_NUM_BINS;
    for (int i = threadIdx.x; i < FSTAT_FINE_NUM_BINS; i += blockDim.x) {
        if (acc[i] != 0ULL) {
            atomicAdd(&out[i], acc[i]);
        }
    }
}


/* ===========================================================================
 * EXACT WIDE-INTEGER COMPARE HELPERS (kernel core 2.3.0 epilogue)
 * ===========================================================================
 * The decision epilogue compares exact rationals built from uint64 fine
 * power sums. Cross products reach 128 bits and the threshold compare is
 * a triple product (192 bits), so the comparisons are formed in explicit
 * hi/lo fixed-width arithmetic --- never floating point, never truncated.
 * The Python reference (src/pilot_proxy/fine_decision.py) performs the
 * same comparisons with arbitrary-precision integers; the results are
 * identical because both are exact.
 */

__device__ __forceinline__
unsigned long long fstat_umul64hi_(unsigned long long a, unsigned long long b)
{
    return __umul64hi(a, b);
}

/**
 * @brief Exact rational comparison: is na/da < nb/db (da, db > 0)?
 */
__device__ __forceinline__
int fstat_frac_less(unsigned long long na, unsigned long long da,
                    unsigned long long nb, unsigned long long db)
{
    const unsigned long long l1 = na * db;
    const unsigned long long h1 = fstat_umul64hi_(na, db);
    const unsigned long long l2 = nb * da;
    const unsigned long long h2 = fstat_umul64hi_(nb, da);
    return (h1 < h2) || (h1 == h2 && l1 < l2);
}

/**
 * @brief 128-bit value (hi:lo) times 64-bit -> exact 192-bit (r2:r1:r0).
 */
__device__ __forceinline__
void fstat_mul_128_by_64(unsigned long long hi, unsigned long long lo,
                         unsigned long long c,
                         unsigned long long* r2, unsigned long long* r1,
                         unsigned long long* r0)
{
    const unsigned long long lo_c_lo = lo * c;
    const unsigned long long lo_c_hi = fstat_umul64hi_(lo, c);
    const unsigned long long hi_c_lo = hi * c;
    const unsigned long long hi_c_hi = fstat_umul64hi_(hi, c);
    const unsigned long long mid = lo_c_hi + hi_c_lo;
    *r0 = lo_c_lo;
    *r1 = mid;
    *r2 = hi_c_hi + (mid < lo_c_hi ? 1ULL : 0ULL);
}

/**
 * @brief Exact triple-product comparison: is a*b*c > d*e*f?
 *
 * All operands are 64-bit; each side is formed exactly in 192 bits.
 */
__device__ __forceinline__
int fstat_triple_greater(unsigned long long a, unsigned long long b,
                         unsigned long long c,
                         unsigned long long d, unsigned long long e,
                         unsigned long long f)
{
    unsigned long long x2, x1, x0, y2, y1, y0;
    fstat_mul_128_by_64(fstat_umul64hi_(a, b), a * b, c, &x2, &x1, &x0);
    fstat_mul_128_by_64(fstat_umul64hi_(d, e), d * e, f, &y2, &y1, &y0);
    if (x2 != y2) return x2 > y2;
    if (x1 != y1) return x1 > y1;
    return x0 > y0;
}

/* ===========================================================================
 * FUSED FINE KERNEL (kernel cores 2.2.0 / 2.3.0)
 * ===========================================================================
 * One launch from packed samples to exact fine and coarse power sums: the
 * fused form recorded in docs/DESIGN_DECISIONS.md ("one solid kernel").
 * Block-per-stream: each block computes its stream's 3 x 128 row sums into
 * shared memory (never materialized to global unless the debug tap is
 * bound), accumulates the exact coarse marginals from the same values ---
 * making the bit-exact marginal identity an internal property of the
 * launch --- and runs the frozen fxfft256 v1 transform in place, with
 * cooperative butterflies (identical arithmetic to fstat_fxfft256_device;
 * stage order preserved, butterflies within a stage write disjoint pairs,
 * so the produced bits are identical for any thread schedule).
 *
 * Acceptance is bit-equality with the composed path
 * (RowSums_I32 -> FinePowers_U64, plus Powers_U64) --- enforced by
 * tests/kernel/test_fused_fine_gpu.py.
 *
 * Kernel core 2.3.0 adds the decision epilogue (MaskOut != NULL): each
 * block's thread 0 increments a per-batch-entry completion counter (the
 * MaskOut element itself, zeroed by the API entry) after a
 * __threadfence; the last-arriving block for a batch entry re-reads the
 * finalized FinePowers sums and forms the frozen fine decision v1
 * (src/pilot_proxy/fine_decision.py): rank-based null-bulk CFAR over
 * the bundle's 256-bit bulk mask, designated-set compare with a Q16
 * multiplier, exact 128/192-bit integer comparisons, then overwrites
 * the counter with the mask bit (1 = reject; degenerate frames forced
 * 0). The epilogue is deterministic because it reads order-independent
 * exact integer sums and rank selection is by value. With
 * MaskOut == NULL the added path is not executed and the kernel is
 * bit-identical to core 2.2.0. PowerTerms may be NULL (2.3.0 entry)
 * to skip the coarse-marginal accumulation entirely.
 */

__global__ void
kernel_fused_fine(
    const InputType* __restrict__ X,
    const InputType* __restrict__ W,
    const int*       __restrict__ W_Lanes,
    unsigned long long* __restrict__ FinePowers,
    unsigned long long* __restrict__ PowerTerms,
    int*             __restrict__ RowSumsTap,
    int*             __restrict__ MaskOut,
    int anchor_bin,
    int designated_half_width,
    unsigned long long bulk_mask_w0,
    unsigned long long bulk_mask_w1,
    unsigned long long bulk_mask_w2,
    unsigned long long bulk_mask_w3,
    int cfar_rank,
    unsigned long long multiplier_q16,
    int num_streams,
    int detector_rows_per_block,
    int batch)
{
    #if FSTAT_USE_DP4A
    (void)W;
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
    (void)W_Lanes;
    #endif
    #else
    (void)W_Lanes;
    #endif

    const int s = blockIdx.x;
    const int b = blockIdx.y;
    if (s >= num_streams || b >= batch) {
        return;
    }

    const int tid = threadIdx.x;

    __shared__ int s_z[FSTAT_NUM_WEIGHT_TERMS][FSTAT_FINE_WINDOWS_PER_STREAM][2];
    __shared__ int s_fft[FSTAT_FINE_NUM_BINS][2];
    __shared__ long long s_part[FSTAT_BLOCK_THREADS];

    #if FSTAT_USE_DP4A && FSTAT_USE_SHARED_WEIGHT_LANES && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    const int* weight_lanes = W_Lanes;
    __shared__ int shared_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
    for (int idx = tid; idx < FSTAT_WEIGHT_LANE_COUNT; idx += blockDim.x) {
        shared_weight_lanes[idx] = W_Lanes[idx];
    }
    __syncthreads();
    weight_lanes = shared_weight_lanes;
    #elif FSTAT_USE_DP4A
    const int* weight_lanes = W_Lanes;
    #endif

    /* Phase 1: row sums for this stream's windows (each thread owns whole
     * windows; no cross-thread reduction), accumulating the exact per-thread
     * marginal partial from the same registers.
     *
     * The window axis is walked with a thread-stride loop, matching the tap
     * loop of the staged kernels and the RowSumsTap loop below. The previous
     * form derived a per-thread window count by integer division, which
     * silently truncated when the window count was not a multiple of the
     * block size and computed nothing at all when it was smaller. Striding
     * removes both requirements and leaves the emitted bits unchanged: s_z is
     * indexed by window, so which thread produces a window does not affect
     * shared-memory contents, and the marginal partials are reduced by integer
     * addition, which is associative and commutative. */
    long long part[FSTAT_NUM_WEIGHT_TERMS];
    #pragma unroll
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        part[n] = 0LL;
    }

    const size_t batch_stride =
        static_cast<size_t>(detector_rows_per_block) * FSTAT_DETECTOR_WINDOW_SAMPLES;
    const InputType* Xb = X + batch_stride * static_cast<size_t>(b);

    for (int w = tid; w < FSTAT_FINE_WINDOWS_PER_STREAM; w += blockDim.x) {
        const int m = s * FSTAT_FINE_WINDOWS_PER_STREAM + w;  /* detector row */

        int2 dot[FSTAT_NUM_WEIGHT_TERMS];
        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            dot[n] = make_int2(0, 0);
        }

        #if FSTAT_USE_DP4A
        for (int pair = 0; pair < FSTAT_DP4A_TAP_PAIRS; ++pair) {
            const int k0 = 2 * pair;
            int a_re;
            int a_im;
            unpack_two_complex_bytes_to_dp4a_lanes(
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0],
                Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k0 + 1],
                a_re,
                a_im);
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                const int w_lane =
                    load_weight_lane(weight_lanes, n * FSTAT_DP4A_TAP_PAIRS + pair);
                dot[n].x = __dp4a(a_re, w_lane, dot[n].x);
                dot[n].y = __dp4a(a_im, w_lane, dot[n].y);
            }
        }
        #else
        for (int k = 0; k < FSTAT_DETECTOR_WINDOW_SAMPLES; ++k) {
            short2 x_val = unpack_sample(Xb[m * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
            #pragma unroll
            for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
                short2 w_val = unpack_sample(W[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k]);
                dot[n] = complex_add(dot[n], complex_multiply(x_val, w_val));
            }
        }
        #endif

        #pragma unroll
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            s_z[n][w][0] = dot[n].x;
            s_z[n][w][1] = dot[n].y;
            const long long zr = static_cast<long long>(dot[n].x);
            const long long zi = static_cast<long long>(dot[n].y);
            part[n] += zr * zr + zi * zi;
        }
    }
    __syncthreads();

    /* Phase 2: exact coarse marginals (the identity is internal: same
     * shared values the FFT consumes), plus the optional row-sum tap.
     * PowerTerms == NULL (uniform across the block) skips the stage. */
    if (PowerTerms != NULL) {
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            s_part[tid] = part[n];
            __syncthreads();
            if (tid == 0) {
                long long acc = 0LL;
                for (int i = 0; i < static_cast<int>(blockDim.x); ++i) {
                    acc += s_part[i];
                }
                atomicAdd(&PowerTerms[static_cast<size_t>(b) * FSTAT_NUM_WEIGHT_TERMS + n],
                          static_cast<unsigned long long>(acc));
            }
            __syncthreads();
        }
    }
    if (RowSumsTap != NULL) {
        int* Out = RowSumsTap
            + static_cast<size_t>(b) * FSTAT_NUM_WEIGHT_TERMS
                  * static_cast<size_t>(detector_rows_per_block) * 2;
        for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
            for (int w = tid; w < FSTAT_FINE_WINDOWS_PER_STREAM; w += blockDim.x) {
                const int m = s * FSTAT_FINE_WINDOWS_PER_STREAM + w;
                Out[(static_cast<size_t>(n) * detector_rows_per_block + m) * 2 + 0] =
                    s_z[n][w][0];
                Out[(static_cast<size_t>(n) * detector_rows_per_block + m) * 2 + 1] =
                    s_z[n][w][1];
            }
        }
    }

    /* Phase 3: frozen fxfft256 v1 per term, cooperative butterflies. */
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        __syncthreads();
        for (int i = tid; i < FSTAT_FINE_NUM_BINS; i += blockDim.x) {
            const unsigned src = fstat_fxfft_bitrev(static_cast<unsigned>(i));
            if (src < FSTAT_FINE_WINDOWS_PER_STREAM) {
                s_fft[i][0] = s_z[n][src][0];
                s_fft[i][1] = s_z[n][src][1];
            } else {
                s_fft[i][0] = 0;
                s_fft[i][1] = 0;
            }
        }
        __syncthreads();
        for (unsigned stage = 1; stage <= (unsigned)FSTAT_FINE_LOG2; ++stage) {
            const unsigned m2 = 1u << stage;
            const unsigned half = m2 >> 1;
            for (unsigned bf = tid; bf < FSTAT_FINE_NUM_BINS / 2u; bf += blockDim.x) {
                const unsigned g = bf / half;
                const unsigned t = bf % half;
                const unsigned i0 = g * m2 + t;
                const unsigned i1 = i0 + half;
                const int* tw = fstat_fxfft_twiddle_q15[t << ((unsigned)FX_MASTER_LOG2 - stage)];
                const long long c = tw[0];
                const long long sn = tw[1];
                const long long br = s_fft[i1][0];
                const long long bi = s_fft[i1][1];
                const int tr = fstat_fxfft_round15(br * c - bi * sn);
                const int ti = fstat_fxfft_round15(bi * c + br * sn);
                const int ar = s_fft[i0][0];
                const int ai = s_fft[i0][1];
                s_fft[i0][0] = ar + tr;
                s_fft[i0][1] = ai + ti;
                s_fft[i1][0] = ar - tr;
                s_fft[i1][1] = ai - ti;
            }
            __syncthreads();
        }
        unsigned long long* fine = FinePowers
            + (static_cast<size_t>(b) * FSTAT_NUM_WEIGHT_TERMS + n)
                  * FSTAT_FINE_NUM_BINS;
        for (int i = tid; i < FSTAT_FINE_NUM_BINS; i += blockDim.x) {
            const long long re = s_fft[i][0];
            const long long im = s_fft[i][1];
            atomicAdd(&fine[i], static_cast<unsigned long long>(re * re + im * im));
        }
    }

    /* Phase 4 (kernel core 2.3.0): decision epilogue, last block only. */
    if (MaskOut == NULL) {
        return;
    }

    /* Completion counter: MaskOut[b] itself counts arrivals (the API
     * entry zeroes it); the last-arriving block overwrites it with the
     * decision, so no separate scratch allocation exists. The
     * threadfence-then-atomic pattern makes every earlier FinePowers
     * atomicAdd visible to the block that observes the final count
     * (CUDA threadFenceReduction pattern). */
    __shared__ int s_last;
    __shared__ int s_int[FSTAT_BLOCK_THREADS];
    __shared__ int s_rank_bin;
    __shared__ int s_nbulk;
    __shared__ unsigned long long s_num[FSTAT_FINE_NUM_BINS];
    __shared__ unsigned long long s_den[FSTAT_FINE_NUM_BINS];
    __threadfence();
    if (tid == 0) {
        const int prior = atomicAdd(&MaskOut[b], 1);
        s_last = (prior == num_streams - 1) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    /* Exact rational per bin: num = 2 S_target, den = S_lo + S_up.
     * Volatile loads bypass L1 so the finalized device-wide atomic sums
     * are read, not a stale local copy. Deployed magnitudes satisfy
     * S < 2^56 (Parseval over the 14336 row-sum bound), so 2 S and the
     * downstream 192-bit products are exact. */
    {
        const unsigned long long* base = FinePowers
            + static_cast<size_t>(b) * FSTAT_NUM_WEIGHT_TERMS
                  * FSTAT_FINE_NUM_BINS;
        for (int i = tid; i < FSTAT_FINE_NUM_BINS; i += blockDim.x) {
            const unsigned long long st =
                *(volatile const unsigned long long*)&base[i];
            const unsigned long long sl =
                *(volatile const unsigned long long*)
                    &base[FSTAT_FINE_NUM_BINS + i];
            const unsigned long long su =
                *(volatile const unsigned long long*)
                    &base[2 * FSTAT_FINE_NUM_BINS + i];
            s_num[i] = 2ULL * st;
            s_den[i] = sl + su;
        }
    }
    __syncthreads();

    const unsigned long long bulk_words[4] = {
        bulk_mask_w0, bulk_mask_w1, bulk_mask_w2, bulk_mask_w3};
    #define FSTAT_BULK_BIT(i) \
        ((bulk_words[(i) >> 6] >> ((i) & 63)) & 1ULL)

    /* Usable bulk census: mask bit set and den > 0. */
    {
        int cnt = 0;
        for (int i = tid; i < FSTAT_FINE_NUM_BINS; i += blockDim.x) {
            if (FSTAT_BULK_BIT(i) && s_den[i] > 0ULL) {
                cnt++;
            }
        }
        s_int[tid] = cnt;
    }
    __syncthreads();
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < static_cast<int>(blockDim.x); ++i) {
            total += s_int[i];
        }
        s_nbulk = total;
    }
    __syncthreads();
    if (s_nbulk <= cfar_rank) {
        /* Degenerate bulk for the requested rank: invalid frame. */
        if (tid == 0) {
            MaskOut[b] = 0;
        }
        return;
    }

    /* Rank selection by counting (value-unique under ties; the lowest
     * qualifying bin is the deterministic representative, and any
     * representative of a tied value yields the same decision). */
    {
        int best = FSTAT_FINE_NUM_BINS;
        for (int i = tid; i < FSTAT_FINE_NUM_BINS; i += blockDim.x) {
            if (!FSTAT_BULK_BIT(i) || s_den[i] == 0ULL) {
                continue;
            }
            int c_lt = 0;
            int c_eq = 0;
            for (int j = 0; j < FSTAT_FINE_NUM_BINS; ++j) {
                if (!FSTAT_BULK_BIT(j) || s_den[j] == 0ULL) {
                    continue;
                }
                if (fstat_frac_less(s_num[j], s_den[j],
                                    s_num[i], s_den[i])) {
                    c_lt++;
                } else if (!fstat_frac_less(s_num[i], s_den[i],
                                            s_num[j], s_den[j])) {
                    c_eq++;
                }
            }
            if (c_lt <= cfar_rank && cfar_rank < c_lt + c_eq && i < best) {
                best = i;
            }
        }
        s_int[tid] = best;
    }
    __syncthreads();
    if (tid == 0) {
        int best = FSTAT_FINE_NUM_BINS;
        for (int i = 0; i < static_cast<int>(blockDim.x); ++i) {
            if (s_int[i] < best) {
                best = s_int[i];
            }
        }
        s_rank_bin = best;
    }
    __syncthreads();

    /* Designated-set compare: F2[bin] > (multiplier_q16 / 2^16) * F2_r
     * as num[bin] * 2^16 * den_r > multiplier_q16 * num_r * den[bin].
     * Degenerate denominators can never fire (zero-reference forced 0,
     * per bin). */
    {
        const unsigned long long num_r = s_num[s_rank_bin];
        const unsigned long long den_r = s_den[s_rank_bin];
        const int set_size = 2 * designated_half_width + 1;
        int fired = 0;
        for (int t = tid; t < set_size; t += blockDim.x) {
            const int bin = (anchor_bin - designated_half_width + t
                             + FSTAT_FINE_NUM_BINS)
                            & (FSTAT_FINE_NUM_BINS - 1);
            if (s_den[bin] > 0ULL
                && fstat_triple_greater(
                       s_num[bin], den_r, 1ULL << 16,
                       multiplier_q16, num_r, s_den[bin])) {
                fired = 1;
            }
        }
        s_int[tid] = fired;
    }
    __syncthreads();
    if (tid == 0) {
        int any = 0;
        for (int i = 0; i < static_cast<int>(blockDim.x); ++i) {
            any |= s_int[i];
        }
        MaskOut[b] = any;
    }
    #undef FSTAT_BULK_BIT
}

/* ===========================================================================
 * INTERNAL DATA STRUCTURES
 * ===========================================================================*/

/**
 * @brief Opaque handle for F-statistic computation state.
 */
struct FStatHandle {
    int        detector_rows_per_block; ///< Rows in detector-matrix view
    int        batch;      ///< Batch size (number of independent blocks)
    const InputType* d_in; ///< Device input pointer (external)
    float*     d_out;      ///< Device output pointer (external)
    unsigned long long* d_power_scratch; ///< Internal integer power sums [num_weight_terms per block]
    InputType* d_weights;  ///< Scalar-path weights [num_weight_terms x detector_window_samples]
    int*       d_weight_lanes; ///< DP4A-path packed weight lanes [num_weight_terms x tap_pairs]
    bool       weights_cached; ///< Host cache validity for avoiding repeated uploads
    InputType  h_weight_cache[FSTAT_WEIGHT_COUNT]; ///< Last packed weights supplied by caller
};

#if FSTAT_USE_DP4A
static int host_sign_extend_i4(int x)
{
    x &= FSTAT_INT4_MASK;
    return (x ^ FSTAT_INT4_SIGN_BIT) - FSTAT_INT4_SIGN_BIT;
}

static int host_pack4_i8(int a0, int a1, int a2, int a3)
{
    unsigned int u0 = static_cast<unsigned char>(static_cast<signed char>(a0));
    unsigned int u1 = static_cast<unsigned char>(static_cast<signed char>(a1));
    unsigned int u2 = static_cast<unsigned char>(static_cast<signed char>(a2));
    unsigned int u3 = static_cast<unsigned char>(static_cast<signed char>(a3));
    return static_cast<int>(
        u0
        | (u1 << FSTAT_DP4A_LANE1_SHIFT)
        | (u2 << FSTAT_DP4A_LANE2_SHIFT)
        | (u3 << FSTAT_DP4A_LANE3_SHIFT));
}

static void host_unpack_i4_complex(InputType packed, int& real, int& imag)
{
    const int byte = static_cast<int>(static_cast<unsigned char>(packed));
    real = host_sign_extend_i4(byte >> FSTAT_INT4_COMPONENT_BITS);
    imag = host_sign_extend_i4(byte);
}

static void prepack_weight_lanes(
    const InputType* w_in,
    int* h_weight_lanes)
{
    for (int n = 0; n < FSTAT_NUM_WEIGHT_TERMS; ++n) {
        for (int pair = 0; pair < FSTAT_DP4A_TAP_PAIRS; ++pair) {
            const int k0 = 2 * pair;
            const int k1 = k0 + 1;

            int w0r;
            int w0i;
            int w1r;
            int w1i;
            host_unpack_i4_complex(w_in[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k0], w0r, w0i);
            host_unpack_i4_complex(w_in[n * FSTAT_DETECTOR_WINDOW_SAMPLES + k1], w1r, w1i);

            h_weight_lanes[n * FSTAT_DP4A_TAP_PAIRS + pair] =
                host_pack4_i8(w0r, w0i, w1r, w1i);
        }
    }
}
#endif

static bool fstat_upload_weights(FStatHandle* h, const InputType* w_in)
{
    if (!h) {
        record_api_error("handle is null.");
        return false;
    }
    if (!w_in) {
        record_api_error("weight pointer is null.");
        return false;
    }

    #if FSTAT_USE_DP4A && FSTAT_USE_CONSTANT_WEIGHT_LANES
    // Constant memory is module-global, not per-handle. Always refresh it
    // before launches to avoid cross-handle stale weight-lane state.
    const bool weights_changed = true;
    #else
    const bool weights_changed =
        !h->weights_cached ||
        (std::memcmp(h->h_weight_cache, w_in, FSTAT_WEIGHT_BYTES) != 0);
    #endif

    if (weights_changed) {
        h->weights_cached = false;

    #if FSTAT_USE_DP4A
        int h_weight_lanes[FSTAT_WEIGHT_LANE_COUNT];
        prepack_weight_lanes(w_in, h_weight_lanes);
    #if FSTAT_USE_CONSTANT_WEIGHT_LANES
        CUDA_CHECK_BOOL(cudaMemcpyToSymbol(
            c_weight_lanes,
            h_weight_lanes,
            FSTAT_WEIGHT_LANE_COUNT * sizeof(int)));
    #else
        CUDA_CHECK_BOOL(cudaMemcpy(
            h->d_weight_lanes,
            h_weight_lanes,
            FSTAT_WEIGHT_LANE_COUNT * sizeof(int),
            cudaMemcpyHostToDevice));
    #endif
    #else
        CUDA_CHECK_BOOL(cudaMemcpy(
            h->d_weights,
            w_in,
            FSTAT_WEIGHT_BYTES,
            cudaMemcpyHostToDevice));
    #endif

        std::memcpy(h->h_weight_cache, w_in, FSTAT_WEIGHT_BYTES);
        h->weights_cached = true;
    }
    return true;
}

static bool fstat_accumulate(FStatHandle* h, const InputType* w_in)
{
    if (!fstat_upload_weights(h, w_in)) {
        return false;
    }

    // Clear integer power scratch buffer
    CUDA_CHECK_BOOL(cudaMemset(
        h->d_power_scratch,
        0,
        h->batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long)));

    // Compute grid dimensions
    int grid_size = (h->detector_rows_per_block + FSTAT_BLOCK_THREADS - 1) / FSTAT_BLOCK_THREADS;
    if (grid_size > FSTAT_GRID_MAX_BLOCKS) {
        grid_size = FSTAT_GRID_MAX_BLOCKS;
    }

    if (h->batch <= 1) {
        kernel_accumulate_power<<<grid_size, FSTAT_BLOCK_THREADS>>>(
            h->d_in,
            h->d_weights,
            h->d_weight_lanes,
            h->d_power_scratch,
            h->detector_rows_per_block
        );
    } else {
        dim3 grid(grid_size, h->batch, 1);
        kernel_accumulate_power_batched<<<grid, FSTAT_BLOCK_THREADS>>>(
            h->d_in,
            h->d_weights,
            h->d_weight_lanes,
            h->d_power_scratch,
            h->detector_rows_per_block,
            h->batch
        );
    }
    CUDA_CHECK_LAST_BOOL();
    CUDA_CHECK_SYNC_BOOL();
    return true;
}

static bool fstat_write_f_statistic(FStatHandle* h)
{
    if (!h) {
        record_api_error("handle is null.");
        return false;
    }
    if (h->d_out == nullptr) {
        record_api_error("output pointer is null for floating F-statistic output.");
        return false;
    }

    if (h->batch <= 1) {
        kernel_compute_f_statistic<<<1, 1>>>(
            h->d_power_scratch, h->d_out
        );
    } else {
        const int threads = FSTAT_OUTPUT_KERNEL_THREADS;
        const int blocks = (h->batch + threads - 1) / threads;
        kernel_compute_f_statistic_batched<<<blocks, threads>>>(
            h->d_power_scratch, h->d_out, h->batch
        );
    }
    CUDA_CHECK_LAST_BOOL();
    CUDA_CHECK_SYNC_BOOL();
    return true;
}

static bool fstat_write_numden_mask_rational(
    FStatHandle* h,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out,
    unsigned int* d_rational_overflow_count)
{
    if (!h) {
        record_api_error("handle is null.");
        return false;
    }
    if (d_num_out == nullptr) {
        record_api_error("numerator output pointer is null.");
        return false;
    }
    if (d_den_out == nullptr) {
        record_api_error("denominator output pointer is null.");
        return false;
    }
    if (d_mask_out == nullptr) {
        record_api_error("mask output pointer is null.");
        return false;
    }
    if (threshold_half_den == 0ULL) {
        record_api_error("threshold_half_den must be nonzero.");
        return false;
    }
    if (d_rational_overflow_count != nullptr) {
        CUDA_CHECK_BOOL(cudaMemset(
            d_rational_overflow_count,
            0,
            sizeof(unsigned int)));
    }

    const int threads = FSTAT_OUTPUT_KERNEL_THREADS;
    const int blocks = (h->batch + threads - 1) / threads;
    kernel_write_num_den_mask_threshold_half_rational<<<blocks, threads>>>(
        h->d_power_scratch,
        d_num_out,
        d_den_out,
        d_mask_out,
        threshold_half_num,
        threshold_half_den,
        h->batch,
        d_rational_overflow_count);
    CUDA_CHECK_LAST_BOOL();
    CUDA_CHECK_SYNC_BOOL();
    return true;
}

static FStatHandle* fstat_create(
    int detector_rows_per_block,
    int batch,
    const InputType* d_in,
    float* d_out)
{
    if (detector_rows_per_block <= 0) {
        record_api_error("detector_rows_per_block must be positive.");
        return nullptr;
    }
    if (batch < 1) {
        record_api_error("batch must be at least one.");
        return nullptr;
    }
    if (d_in == nullptr) {
        record_api_error("input pointer is null.");
        return nullptr;
    }

    int device = 0;
    if (!record_cuda_error(cudaGetDevice(&device), __FILE__, __LINE__)) {
        return nullptr;
    }
    cudaDeviceProp prop;
    if (!record_cuda_error(cudaGetDeviceProperties(&prop, device), __FILE__, __LINE__)) {
        return nullptr;
    }
    if (batch > prop.maxGridSize[1]) {
        record_api_error("batch exceeds CUDA grid.y limit.");
        return nullptr;
    }

    FStatHandle* h = new (std::nothrow) FStatHandle;
    if (!h) {
        record_api_error("host allocation failed.");
        return nullptr;
    }

    h->detector_rows_per_block = detector_rows_per_block;
    h->batch     = batch;
    h->d_in      = d_in;
    h->d_out     = d_out;
    h->d_power_scratch = nullptr;
    h->d_weights = nullptr;
    h->d_weight_lanes = nullptr;
    h->weights_cached = false;

    if (!record_cuda_error(cudaMalloc(
        &h->d_power_scratch,
        batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long)), __FILE__,
    __LINE__)) {
        delete h;
        return nullptr;
    }
    if (!record_cuda_error(cudaMemset(
        h->d_power_scratch,
        0,
        batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long)), __FILE__,
    __LINE__)) {
        cudaFree(h->d_power_scratch);
        delete h;
        return nullptr;
    }
    #if FSTAT_USE_DP4A && !FSTAT_USE_CONSTANT_WEIGHT_LANES
    if (!record_cuda_error(cudaMalloc(
            &h->d_weight_lanes,
            FSTAT_WEIGHT_LANE_COUNT * sizeof(int)),
            __FILE__,
            __LINE__)) {
        cudaFree(h->d_power_scratch);
        delete h;
        return nullptr;
    }
    #elif FSTAT_USE_DP4A && FSTAT_USE_CONSTANT_WEIGHT_LANES
    // No per-handle weight storage is required. Weight lanes are copied into
    // module-global constant memory before each launch.
    #else
    if (!record_cuda_error(cudaMalloc(
            &h->d_weights,
            FSTAT_WEIGHT_BYTES),
            __FILE__,
            __LINE__)) {
        cudaFree(h->d_power_scratch);
        delete h;
        return nullptr;
    }
    #endif

    return h;
}

/* ===========================================================================
 * C API IMPLEMENTATION
 * ===========================================================================*/

extern "C" {

void FStat_GetSpecs(
    int* detector_window_samples,
    int* num_weight_terms,
    int* sample_bits_per_component,
    int* reference_offset_bins)
{
    clear_last_error();
    if (detector_window_samples) {
        *detector_window_samples = FSTAT_DETECTOR_WINDOW_SAMPLES;
    }
    if (num_weight_terms) {
        *num_weight_terms = FSTAT_NUM_WEIGHT_TERMS;
    }
    if (sample_bits_per_component) {
        *sample_bits_per_component = FSTAT_SAMPLE_BITS_PER_COMPONENT;
    }
    if (reference_offset_bins) {
        *reference_offset_bins = FSTAT_REFERENCE_BIN_OFFSET;
    }
}

void FStat_GetFeatures(
    int* use_dp4a,
    int* use_uint64_power_accumulation,
    int* block_threads)
{
    clear_last_error();
    if (use_dp4a) {
        *use_dp4a = FSTAT_USE_DP4A;
    }
    if (use_uint64_power_accumulation) {
        *use_uint64_power_accumulation = FSTAT_USE_UINT64_POWER_ACCUMULATION;
    }
    if (block_threads) {
        *block_threads = FSTAT_BLOCK_THREADS;
    }
}

void FStat_GetOptimizationFeatures(
    int* use_constant_weight_lanes,
    int* use_shared_weight_lanes,
    int* grid_max_blocks)
{
    clear_last_error();
    if (use_constant_weight_lanes) {
        *use_constant_weight_lanes = FSTAT_USE_CONSTANT_WEIGHT_LANES;
    }
    if (use_shared_weight_lanes) {
        *use_shared_weight_lanes = FSTAT_USE_SHARED_WEIGHT_LANES;
    }
    if (grid_max_blocks) {
        *grid_max_blocks = FSTAT_GRID_MAX_BLOCKS;
    }
}

void FStat_GetVersion(int* major, int* minor, int* patch)
{
    clear_last_error();
    if (major) {
        *major = FSTAT_CORE_VERSION_MAJOR;
    }
    if (minor) {
        *minor = FSTAT_CORE_VERSION_MINOR;
    }
    if (patch) {
        *patch = FSTAT_CORE_VERSION_PATCH;
    }
}

void* FStat_Create(
    const InputType* d_in,
    float* d_out,
    int detector_rows_per_block)
{
    clear_last_error();
    return static_cast<void*>(fstat_create(detector_rows_per_block, 1, d_in, d_out));
}

const char* FStat_LastError(void)
{
    return g_last_error;
}

void* FStat_Create_Batch(
    const InputType* d_in,
    float* d_out,
    int detector_rows_per_block,
    int batch)
{
    clear_last_error();
    return static_cast<void*>(fstat_create(detector_rows_per_block, batch, d_in, d_out));
}

void FStat_Destroy(void* handle)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) return;

    bool ok = true;

    if (h->d_power_scratch) {
        ok &= record_cuda_error(cudaFree(h->d_power_scratch), __FILE__, __LINE__);
        h->d_power_scratch = nullptr;
    }
    if (h->d_weights) {
        ok &= record_cuda_error(cudaFree(h->d_weights), __FILE__, __LINE__);
        h->d_weights = nullptr;
    }
    if (h->d_weight_lanes) {
        ok &= record_cuda_error(cudaFree(h->d_weight_lanes), __FILE__, __LINE__);
        h->d_weight_lanes = nullptr;
    }
    delete h;
    (void)ok;
}

void FStat_Compute_DiagnosticFloat(void* handle, const InputType* w_in)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;
    if (!fstat_write_f_statistic(h)) return;
}

void FStat_Compute_NumDen_Mask_RationalHalf(
    void* handle,
    const InputType* w_in,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_num_out == nullptr) {
        record_api_error("numerator output pointer is null.");
        return;
    }
    if (d_den_out == nullptr) {
        record_api_error("denominator output pointer is null.");
        return;
    }
    if (d_mask_out == nullptr) {
        record_api_error("mask output pointer is null.");
        return;
    }
    if (threshold_half_den == 0ULL) {
        record_api_error("threshold_half_den must be nonzero.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;
    if (!fstat_write_numden_mask_rational(
        h,
        threshold_half_num,
        threshold_half_den,
        d_num_out,
        d_den_out,
        d_mask_out,
        nullptr)) return;
}

void FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
    void* handle,
    const InputType* w_in,
    unsigned long long threshold_half_num,
    unsigned long long threshold_half_den,
    unsigned long long* d_num_out,
    unsigned long long* d_den_out,
    unsigned char* d_mask_out,
    unsigned int* d_rational_overflow_count)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_num_out == nullptr) {
        record_api_error("numerator output pointer is null.");
        return;
    }
    if (d_den_out == nullptr) {
        record_api_error("denominator output pointer is null.");
        return;
    }
    if (d_mask_out == nullptr) {
        record_api_error("mask output pointer is null.");
        return;
    }
    if (d_rational_overflow_count == nullptr) {
        record_api_error("rational overflow-count pointer is null.");
        return;
    }
    if (threshold_half_den == 0ULL) {
        record_api_error("threshold_half_den must be nonzero.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;
    if (!fstat_write_numden_mask_rational(
        h,
        threshold_half_num,
        threshold_half_den,
        d_num_out,
        d_den_out,
        d_mask_out,
        d_rational_overflow_count)) return;
}

void FStat_Compute_Powers(void* handle, const InputType* w_in)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (h->d_out == nullptr) {
        record_api_error("output pointer is null for float power output.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;

    const int count = h->batch * FSTAT_NUM_WEIGHT_TERMS;
    const int threads = FSTAT_OUTPUT_KERNEL_THREADS;
    const int blocks = (count + threads - 1) / threads;
    kernel_convert_power_terms_to_float<<<blocks, threads>>>(
        h->d_power_scratch,
        h->d_out,
        count);
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();
}

void FStat_Compute_Powers_U64(
    void* handle,
    const InputType* w_in,
    unsigned long long* d_power_out)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_power_out == nullptr) {
        record_api_error("uint64 power output pointer is null.");
        return;
    }

    if (!fstat_accumulate(h, w_in)) return;

    CUDA_CHECK(cudaMemcpy(
        d_power_out,
        h->d_power_scratch,
        h->batch * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long),
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK_SYNC();
}

/**
 * See f_statistic.h for the full layout contract.
 */
void FStat_Compute_RowSums_I32(
    void* handle,
    const InputType* w_in,
    int* d_row_sums_out)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_row_sums_out == nullptr) {
        record_api_error("int32 row-sum output pointer is null.");
        return;
    }

    if (!fstat_upload_weights(h, w_in)) return;

    int grid_size = (h->detector_rows_per_block + FSTAT_BLOCK_THREADS - 1) / FSTAT_BLOCK_THREADS;
    if (grid_size > FSTAT_GRID_MAX_BLOCKS) {
        grid_size = FSTAT_GRID_MAX_BLOCKS;
    }

    if (h->batch <= 1) {
        kernel_accumulate_row_sums<<<grid_size, FSTAT_BLOCK_THREADS>>>(
            h->d_in,
            h->d_weights,
            h->d_weight_lanes,
            d_row_sums_out,
            h->detector_rows_per_block
        );
    } else {
        dim3 grid(grid_size, h->batch, 1);
        kernel_accumulate_row_sums_batched<<<grid, FSTAT_BLOCK_THREADS>>>(
            h->d_in,
            h->d_weights,
            h->d_weight_lanes,
            d_row_sums_out,
            h->detector_rows_per_block,
            h->batch
        );
    }
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();
}

int FStat_Supports_RowSums(void)
{
    return 1;
}



/**
 * See f_statistic.h for the full contract.
 */
void FStat_Compute_FinePowers_U64(
    const int* d_row_sums,
    int num_weight_terms,
    int num_streams,
    int windows_per_stream,
    int batch,
    unsigned long long* d_fine_power_out)
{
    clear_last_error();
    if (d_row_sums == nullptr) {
        record_api_error("row-sum input pointer is null.");
        return;
    }
    if (d_fine_power_out == nullptr) {
        record_api_error("fine power output pointer is null.");
        return;
    }
    if (num_weight_terms != FSTAT_NUM_WEIGHT_TERMS) {
        record_api_error("num_weight_terms must equal the compiled weight-term count.");
        return;
    }
    if (windows_per_stream != FSTAT_FINE_WINDOWS_PER_STREAM) {
        record_api_error(
            "windows_per_stream must equal the frozen fxfft256 transform "
            "length (128).");
        return;
    }
    if (num_streams < 1) {
        record_api_error("num_streams must be >= 1.");
        return;
    }
    if (batch < 1) {
        record_api_error("batch must be >= 1.");
        return;
    }
    if (num_streams > FSTAT_BLOCK_THREADS * FSTAT_GRID_MAX_BLOCKS) {
        record_api_error("num_streams exceeds the launch capacity.");
        return;
    }

    const size_t out_bytes = static_cast<size_t>(batch)
        * FSTAT_NUM_WEIGHT_TERMS * FSTAT_FINE_NUM_BINS
        * sizeof(unsigned long long);
    CUDA_CHECK(cudaMemset(d_fine_power_out, 0, out_bytes));

    const int chunks =
        (num_streams + FSTAT_BLOCK_THREADS - 1) / FSTAT_BLOCK_THREADS;
    dim3 grid(chunks, FSTAT_NUM_WEIGHT_TERMS, batch);
    kernel_fine_powers<<<grid, FSTAT_BLOCK_THREADS>>>(
        d_row_sums,
        d_fine_power_out,
        num_streams,
        batch);
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();
}

int FStat_Supports_FinePowers(void)
{
    return 1;
}

void FStat_GetFineSpecs(
    int* windows_per_stream,
    int* pad_factor,
    int* fine_bins)
{
    if (windows_per_stream) {
        *windows_per_stream = FSTAT_FINE_WINDOWS_PER_STREAM;
    }
    if (pad_factor) {
        *pad_factor = FSTAT_FINE_PAD_FACTOR;
    }
    if (fine_bins) {
        *fine_bins = FSTAT_FINE_NUM_BINS;
    }
}


/**
 * See f_statistic.h for the full contract.
 */
void FStat_Compute_FusedFine_U64(
    void* handle,
    const InputType* w_in,
    unsigned long long* d_fine_power_out,
    unsigned long long* d_power_out,
    int* d_row_sums_out)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_fine_power_out == nullptr) {
        record_api_error("fine power output pointer is null.");
        return;
    }
    if (d_power_out == nullptr) {
        record_api_error("power output pointer is null.");
        return;
    }
    if (h->detector_rows_per_block % FSTAT_FINE_WINDOWS_PER_STREAM != 0) {
        record_api_error(
            "detector_rows_per_block must be a multiple of the frozen "
            "fxfft256 window count (128).");
        return;
    }
    if (FSTAT_FINE_WINDOWS_PER_STREAM % FSTAT_BLOCK_THREADS != 0) {
        record_api_error("block size must divide the window count.");
        return;
    }

    if (!fstat_upload_weights(h, w_in)) return;

    const int num_streams = h->detector_rows_per_block / FSTAT_FINE_WINDOWS_PER_STREAM;
    const size_t fine_bytes = static_cast<size_t>(h->batch)
        * FSTAT_NUM_WEIGHT_TERMS * FSTAT_FINE_NUM_BINS
        * sizeof(unsigned long long);
    const size_t power_bytes = static_cast<size_t>(h->batch)
        * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long);
    CUDA_CHECK(cudaMemset(d_fine_power_out, 0, fine_bytes));
    CUDA_CHECK(cudaMemset(d_power_out, 0, power_bytes));

    dim3 grid(num_streams, h->batch, 1);
    kernel_fused_fine<<<grid, FSTAT_BLOCK_THREADS>>>(
        h->d_in,
        h->d_weights,
        h->d_weight_lanes,
        d_fine_power_out,
        d_power_out,
        d_row_sums_out,
        NULL,            /* no decision epilogue: core 2.2.0 behavior */
        0, 0, 0ULL, 0ULL, 0ULL, 0ULL, 0, 0ULL,
        num_streams,
        h->detector_rows_per_block,
        h->batch);
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();
}

int FStat_Supports_FusedFine(void)
{
    return 1;
}

/**
 * See f_statistic.h for the full contract.
 */
void FStat_Compute_FusedFineMask_U64(
    void* handle,
    const InputType* w_in,
    int anchor_bin,
    int designated_half_width,
    const unsigned long long* bulk_mask_words,
    int cfar_rank,
    unsigned long long multiplier_q16,
    unsigned long long* d_fine_power_out,
    int* d_mask_out,
    unsigned long long* d_power_out,
    int* d_row_sums_out)
{
    clear_last_error();
    FStatHandle* h = static_cast<FStatHandle*>(handle);
    if (!h) {
        record_api_error("handle is null.");
        return;
    }
    if (d_fine_power_out == nullptr) {
        record_api_error("fine power output pointer is null.");
        return;
    }
    if (d_mask_out == nullptr) {
        record_api_error("mask output pointer is null.");
        return;
    }
    if (bulk_mask_words == nullptr) {
        record_api_error("bulk mask words pointer is null.");
        return;
    }
    if (anchor_bin < 0 || anchor_bin >= FSTAT_FINE_NUM_BINS) {
        record_api_error("anchor_bin must be in [0, 256).");
        return;
    }
    if (designated_half_width < 0
        || designated_half_width >= FSTAT_FINE_NUM_BINS / 2) {
        record_api_error("designated_half_width must be in [0, 128).");
        return;
    }
    if (cfar_rank < 0 || cfar_rank >= FSTAT_FINE_NUM_BINS) {
        record_api_error("cfar_rank must be in [0, 256).");
        return;
    }
    if (h->detector_rows_per_block % FSTAT_FINE_WINDOWS_PER_STREAM != 0) {
        record_api_error(
            "detector_rows_per_block must be a multiple of the frozen "
            "fxfft256 window count (128).");
        return;
    }
    if (FSTAT_FINE_WINDOWS_PER_STREAM % FSTAT_BLOCK_THREADS != 0) {
        record_api_error("block size must divide the window count.");
        return;
    }

    if (!fstat_upload_weights(h, w_in)) return;

    const int num_streams =
        h->detector_rows_per_block / FSTAT_FINE_WINDOWS_PER_STREAM;
    const size_t fine_bytes = static_cast<size_t>(h->batch)
        * FSTAT_NUM_WEIGHT_TERMS * FSTAT_FINE_NUM_BINS
        * sizeof(unsigned long long);
    CUDA_CHECK(cudaMemset(d_fine_power_out, 0, fine_bytes));
    /* The mask buffer doubles as the completion counter and must start
     * at zero for every launch. */
    CUDA_CHECK(cudaMemset(
        d_mask_out, 0, static_cast<size_t>(h->batch) * sizeof(int)));
    if (d_power_out != nullptr) {
        const size_t power_bytes = static_cast<size_t>(h->batch)
            * FSTAT_NUM_WEIGHT_TERMS * sizeof(unsigned long long);
        CUDA_CHECK(cudaMemset(d_power_out, 0, power_bytes));
    }

    dim3 grid(num_streams, h->batch, 1);
    kernel_fused_fine<<<grid, FSTAT_BLOCK_THREADS>>>(
        h->d_in,
        h->d_weights,
        h->d_weight_lanes,
        d_fine_power_out,
        d_power_out,
        d_row_sums_out,
        d_mask_out,
        anchor_bin,
        designated_half_width,
        bulk_mask_words[0],
        bulk_mask_words[1],
        bulk_mask_words[2],
        bulk_mask_words[3],
        cfar_rank,
        multiplier_q16,
        num_streams,
        h->detector_rows_per_block,
        h->batch);
    CUDA_CHECK_LAST();
    CUDA_CHECK_SYNC();
}

int FStat_Supports_FusedFineMask(void)
{
    return 1;
}

} // extern "C"
