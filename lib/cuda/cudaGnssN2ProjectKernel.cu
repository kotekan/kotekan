#include "cudaGnssN2ProjectKernel.hpp"

namespace gnss_cuda {

namespace {

// Basis record (GnssN2Project.hpp): 64-byte header, int32 stations[n], then per slot int32
// freq_id, k, two unused and complex<float> q[k_slot][n]. The slots start 4 n bytes in, so
// their floats are read one at a time (an odd n leaves them 4-byte aligned only).
__device__ inline const uint8_t* rec_slot(const uint8_t* rec, int n, int k_slot, int s) {
    return rec + 64 + 4 * (size_t)n + (size_t)s * (16 + 8 * (size_t)k_slot * n);
}

// Word offset of entry (i, j), i >= j, in one channel (gnss_n2proj::Layout::idx).
__device__ inline size_t widx(int i, int j) {
    const int ihi = i / N2P_BS, jhi = j / N2P_BS, ilo = i % N2P_BS, jlo = j % N2P_BS;
    const size_t b = (size_t)ihi * (ihi + 1) / 2 + jhi;
    return ((b * N2P_BS + ilo) * N2P_BS + jlo) * 2;
}

// gnss_n2proj::to_i32: nearest, ties to even, saturated.
__device__ inline int32_t to_i32(double x) {
    const double r = rint(x);
    if (r > 2147483647.0)
        return 2147483647;
    if (r < -2147483648.0)
        return -2147483647 - 1;
    return (int32_t)r;
}

__global__ void k_gather(const int4* frame, const int* chan, int4* out) {
    constexpr int W4 = N2P_WORDS / 4;
    const int s = blockIdx.y;
    const int4* src = frame + (size_t)chan[s] * W4;
    int4* dst = out + (size_t)s * W4;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < W4; x += gridDim.x * blockDim.x)
        dst[x] = src[x];
}

// One block per (column j, slot), a thread per row i.
__global__ void k_dense_v(const int32_t* frame, const int* chan, float2* V) {
    const int s = blockIdx.y, j = blockIdx.x, i = threadIdx.x;
    const int32_t* f = frame + (size_t)chan[s] * N2P_WORDS;
    const size_t w = widx(max(i, j), min(i, j));
    // V(i, j) = f(i, j) below the diagonal and conj f(j, i) above it; extract() leaves the
    // diagonal conjugated too.
    const float re = f[w], im = i > j ? f[w + 1] : -(float)f[w + 1];
    V[(size_t)s * N2P_INPUTS * N2P_INPUTS + i + (size_t)N2P_INPUTS * j] = make_float2(re, im);
}

// One block per slot, a thread per input.
__global__ void k_embed_q(const uint8_t* rec, int n, int k_slot, int k_ld, float2* Qt) {
    __shared__ int inv[N2P_INPUTS];
    const int s = blockIdx.x, i = threadIdx.x;
    inv[i] = -1;
    __syncthreads();
    const int32_t* st = (const int32_t*)(rec + 64);
    for (int a = i; a < n; a += blockDim.x) {
        const int e = st[a];
        if (e >= 0 && e < N2P_INPUTS)
            inv[e] = a;
    }
    __syncthreads();
    const uint8_t* slot = rec_slot(rec, n, k_slot, s);
    const int k = min(*(const int32_t*)(slot + 4), k_slot);
    const float* q = (const float*)(slot + 16);
    float2* out = Qt + (size_t)s * N2P_INPUTS * k_ld;
    const int a = inv[i];
    for (int c = 0; c < k_ld; ++c) {
        float2 v = make_float2(0.0f, 0.0f);
        if (c < k && a >= 0) {
            const size_t x = 2 * ((size_t)c * n + a);
            v = make_float2(q[x], q[x + 1]);
        }
        out[i + (size_t)N2P_INPUTS * c] = v;
    }
}

// One block per (16x16 lower-triangle block, slot), a thread per entry.
__global__ void k_update(int32_t* frame, const int* chan, const uint8_t* rec, int n, int k_slot,
                         int k, int k_ld, const float2* Qt, const float2* Z) {
    extern __shared__ float2 sm[]; // Qt and Z rows of the block's i and j: 4 x [k][16]
    __shared__ unsigned char is_st[N2P_INPUTS];
    const int s = blockIdx.y, b = blockIdx.x;
    const int t = threadIdx.y * N2P_BS + threadIdx.x;
    if (*(const int32_t*)(rec_slot(rec, n, k_slot, s) + 4) <= 0)
        return; // the whole block: nothing projected on this slot
    if (t < N2P_INPUTS)
        is_st[t] = 0;
    __syncthreads();
    const int32_t* st = (const int32_t*)(rec + 64);
    for (int a = t; a < n; a += N2P_BS * N2P_BS) {
        const int e = st[a];
        if (e >= 0 && e < N2P_INPUTS)
            is_st[e] = 1;
    }
    int ihi = 0;
    while ((ihi + 1) * (ihi + 2) / 2 <= b)
        ++ihi;
    const int jhi = b - ihi * (ihi + 1) / 2;
    float2* sQi = sm;
    float2* sZi = sQi + k * N2P_BS;
    float2* sQj = sZi + k * N2P_BS;
    float2* sZj = sQj + k * N2P_BS;
    const float2* q = Qt + (size_t)s * N2P_INPUTS * k_ld;
    const float2* z = Z + (size_t)s * N2P_INPUTS * k_ld;
    for (int x = t; x < k * N2P_BS; x += N2P_BS * N2P_BS) {
        const int c = x / N2P_BS, r = x % N2P_BS;
        const size_t ci = (size_t)N2P_INPUTS * c;
        sQi[x] = q[ihi * N2P_BS + r + ci];
        sZi[x] = z[ihi * N2P_BS + r + ci];
        sQj[x] = q[jhi * N2P_BS + r + ci];
        sZj[x] = z[jhi * N2P_BS + r + ci];
    }
    __syncthreads();
    const int ilo = threadIdx.y, jlo = threadIdx.x;
    if (ihi == jhi && jlo > ilo)
        return; // the upper half of a diagonal block is not data
    const int i = ihi * N2P_BS + ilo, j = jhi * N2P_BS + jlo;
    if (!is_st[i] && !is_st[j])
        return; // the other inputs' own products stay as they are
    float ur = 0.0f, ui = 0.0f;
    for (int c = 0; c < k; ++c) {
        // Qt_ic conj(Z_jc) + Z_ic conj(Qt_jc)
        const float2 a = sQi[c * N2P_BS + ilo], bz = sZj[c * N2P_BS + jlo];
        const float2 az = sZi[c * N2P_BS + ilo], bq = sQj[c * N2P_BS + jlo];
        ur += a.x * bz.x + a.y * bz.y + az.x * bq.x + az.y * bq.y;
        ui += a.y * bz.x - a.x * bz.y + az.y * bq.x - az.x * bq.y;
    }
    int32_t* f = frame + (size_t)chan[s] * N2P_WORDS;
    const size_t w = widx(i, j);
    f[w] = to_i32((double)f[w] - ur);
    f[w + 1] = i == j ? 0 : to_i32((double)f[w + 1] - ui);
}

} // namespace

void launch_n2proj_gather(const int32_t* frame, const int* chan, int n_slots, int32_t* out,
                          cudaStream_t stream) {
    if (n_slots <= 0)
        return;
    k_gather<<<dim3(18, n_slots), 256, 0, stream>>>((const int4*)frame, chan, (int4*)out);
}

void launch_n2proj_dense(const int32_t* frame, const int* chan, const uint8_t* rec, int n,
                         int k_slot, int n_slots, int k_ld, float2* V, float2* Qt,
                         cudaStream_t stream) {
    if (n_slots <= 0)
        return;
    k_dense_v<<<dim3(N2P_INPUTS, n_slots), N2P_INPUTS, 0, stream>>>(frame, chan, V);
    k_embed_q<<<n_slots, N2P_INPUTS, 0, stream>>>(rec, n, k_slot, k_ld, Qt);
}

cublasStatus_t n2proj_wz(cublasHandle_t handle, int n_slots, int k, int k_ld, const float2* V,
                         const float2* Qt, float2* W, float2* M) {
    if (n_slots <= 0 || k <= 0)
        return CUBLAS_STATUS_SUCCESS;
    const cuComplex one{1.0f, 0.0f}, zero{0.0f, 0.0f}, mhalf{-0.5f, 0.0f};
    const long long sV = (long long)N2P_INPUTS * N2P_INPUTS, sQ = (long long)N2P_INPUTS * k_ld,
                    sM = (long long)k_ld * k_ld;
    const auto* v = (const cuComplex*)V;
    const auto* q = (const cuComplex*)Qt;
    auto* w = (cuComplex*)W;
    auto* m = (cuComplex*)M;
    cublasStatus_t st = cublasCgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N2P_INPUTS, k, N2P_INPUTS, &one, v, N2P_INPUTS, sV, q,
        N2P_INPUTS, sQ, &zero, w, N2P_INPUTS, sQ, n_slots);
    if (st != CUBLAS_STATUS_SUCCESS)
        return st;
    st = cublasCgemmStridedBatched(handle, CUBLAS_OP_C, CUBLAS_OP_N, k, k, N2P_INPUTS, &one, q,
                                   N2P_INPUTS, sQ, w, N2P_INPUTS, sQ, &zero, m, k_ld, sM, n_slots);
    if (st != CUBLAS_STATUS_SUCCESS)
        return st;
    return cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N2P_INPUTS, k, k, &mhalf, q,
                                     N2P_INPUTS, sQ, m, k_ld, sM, &one, w, N2P_INPUTS, sQ, n_slots);
}

void launch_n2proj_update(int32_t* frame, const int* chan, const uint8_t* rec, int n, int k_slot,
                          int n_slots, int k, int k_ld, const float2* Qt, const float2* Z,
                          cudaStream_t stream) {
    if (n_slots <= 0 || k <= 0)
        return;
    const size_t sh = 4 * (size_t)k * N2P_BS * sizeof(float2);
    // Above the default 48 KB with is_st (k >= 96) the kernel must opt in, on the current device.
    if (sh + N2P_INPUTS > 48 * 1024)
        cudaFuncSetAttribute(k_update, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    k_update<<<dim3(N2P_BLOCKS, n_slots), dim3(N2P_BS, N2P_BS), sh, stream>>>(
        frame, chan, rec, n, k_slot, k, k_ld, Qt, Z);
}

} // namespace gnss_cuda
