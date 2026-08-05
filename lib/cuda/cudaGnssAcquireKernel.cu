#include "cudaGnssAcquireKernel.hpp"

#include <cuda_runtime.h>

namespace gnss_cuda {

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// ---------------------------------------------------------------------------------------------
// DIRECT path: one block per (d,q) row, one thread per fine-lag column.
//
// The ramp table is [nc][s_cols] complex = 81 KB at CHORD scale, well past shared memory, so it
// is read from global and relies on L2 (it is reused by every one of the ~1M rows, so it stays
// resident). This path exists as the reference the folded path must reproduce, and to make the
// "is the fold worth it on GPU?" question measurable rather than arguable.
// ---------------------------------------------------------------------------------------------
__global__ void agg_direct_kernel(const float2* __restrict__ P, float* __restrict__ surf,
                                  const float2* __restrict__ ramp, int nc, int s_cols,
                                  long n_rows) {
    const long row = blockIdx.x;
    if (row >= n_rows)
        return;
    const int s = threadIdx.x;
    if (s >= s_cols)
        return;

    const float2* __restrict__ Prow = P + row * (long)nc; // channel is contiguous: coalesced
    float re = 0.0f, im = 0.0f;
    for (int c = 0; c < nc; ++c) {
        const float2 p = Prow[c];
        const float2 r = ramp[(long)c * s_cols + s];
        re += p.x * r.x - p.y * r.y;
        im += p.x * r.y + p.y * r.x;
    }
    surf[row * (long)s_cols + s] += re * re + im * im;
}

// ---------------------------------------------------------------------------------------------
// FOLDED path: one block per (d,q) row.
//
//   1. fold  nc channels into s_cols bins in shared memory   (nc adds, not nc*s_cols MACs)
//   2. one s_cols-point INVERSE DFT of that bin array        (sign e^{+i}, matching the ramp)
//   3. accumulate |.|^2
//
// s_cols must be a power of two (it is sph/fine_step with both powers of two in every CHORD and
// airspy geometry); the host refuses the folded path otherwise rather than approximating.
//
// Radix-2 decimation-in-time: bit-reverse into shared, then log2(s_cols) butterfly stages. With
// blockDim == s_cols, half the threads idle in each stage -- irrelevant next to the global-memory
// traffic this kernel is bound by, and it keeps the load and store phases perfectly coalesced.
// ---------------------------------------------------------------------------------------------
extern __shared__ float2 smem[]; // [s_cols] fold/FFT buffer, then [s_cols/2] twiddles

__global__ void agg_folded_kernel(const float2* __restrict__ P, float* __restrict__ surf,
                                  const int* __restrict__ bin, const float2* __restrict__ twiddle,
                                  int nc, int s_cols, int log2n, long n_rows) {
    const long row = blockIdx.x;
    if (row >= n_rows)
        return;
    const int t = threadIdx.x;

    float2* F = smem;             // [s_cols]
    float2* W = smem + s_cols;    // [s_cols/2]
    for (int i = t; i < s_cols / 2; i += blockDim.x)
        W[i] = twiddle[i];
    for (int i = t; i < s_cols; i += blockDim.x)
        F[i] = make_float2(0.0f, 0.0f);
    __syncthreads();

    // Fold. Channel is contiguous for this row, so the load is coalesced. Bins CAN collide (two
    // covering channels congruent mod s_cols), so this must be an atomic add -- with a comb
    // narrower than s_cols they never do, but "never on this band" is not a guarantee.
    const float2* __restrict__ Prow = P + row * (long)nc;
    for (int c = t; c < nc; c += blockDim.x) {
        const float2 p = Prow[c];
        const int b = bin[c];
        atomicAdd(&F[b].x, p.x);
        atomicAdd(&F[b].y, p.y);
    }
    __syncthreads();

    // Bit-reversal permutation (into registers, then back -- avoids a second shared buffer).
    float2 v = make_float2(0.0f, 0.0f);
    int rev = 0;
    if (t < s_cols) {
        rev = __brev((unsigned)t) >> (32 - log2n);
        v = F[rev];
    }
    __syncthreads();
    if (t < s_cols)
        F[t] = v;
    __syncthreads();

    // Butterflies. twiddle[j * (s_cols/len)] == e^{+2 pi i j / len}: the INVERSE transform,
    // matching the aggregate's e^{+i...} ramp. Getting this sign wrong puts the peak at
    // s_cols - s and reads as an off-by-one indexing bug.
    for (int len = 2; len <= s_cols; len <<= 1) {
        const int half = len >> 1;
        const int k = t;
        if (k < s_cols / 2) {
            const int j = k & (half - 1);
            const int base = (k / half) * len;
            const int i0 = base + j, i1 = i0 + half;
            const float2 w = W[j * (s_cols / len)];
            const float2 a = F[i0];
            const float2 b = cmul(F[i1], w);
            F[i0] = cadd(a, b);
            F[i1] = csub(a, b);
        }
        __syncthreads();
    }

    if (t < s_cols) {
        const float2 d = F[t];
        surf[row * (long)s_cols + t] += d.x * d.x + d.y * d.y;
    }
}

void launch_aggregate(const float2* P, float* surf, const float2* ramp, const int* bin,
                      const float2* twiddle, int nc, int n_dop, int Mp, int s_cols, bool folded,
                      cudaStream_t stream) {
    const long n_rows = (long)n_dop * Mp;
    if (n_rows <= 0 || nc <= 0 || s_cols <= 0)
        return;
    const int threads = s_cols;
    if (!folded) {
        agg_direct_kernel<<<(unsigned)n_rows, threads, 0, stream>>>(P, surf, ramp, nc, s_cols,
                                                                    n_rows);
        return;
    }
    int log2n = 0;
    while ((1 << log2n) < s_cols)
        ++log2n;
    const size_t shm = (size_t)(s_cols + s_cols / 2) * sizeof(float2);
    agg_folded_kernel<<<(unsigned)n_rows, threads, shm, stream>>>(P, surf, bin, twiddle, nc,
                                                                  s_cols, log2n, n_rows);
}

// ---------------------------------------------------------------------------------------------
// Peak + mean, on device.
//
// channelized_peak takes the FIRST STRICT maximum and normalizes by the surface mean, so this
// must reproduce both: ties resolve to the LOWEST flat index, and the sum is accumulated in
// double (128M cells of O(100) in float would lose the mean to round-off, and the mean is the
// denominator of every reported SNR).
// ---------------------------------------------------------------------------------------------
static constexpr int PEAK_THREADS = 256;
// Cells streamed per thread before the block reduction runs. At 32 this kernel managed only
// 195 GB/s on an L40S against the aggregate's 671: a grid-stride loop of 32 iterations followed
// by 8 __syncthreads() reduction stages spends as much time reducing as reading. Raising it
// amortizes the reduction over 8x more streaming and costs nothing but a smaller grid, which is
// still ~1900 blocks over 142 SMs.
static constexpr int PEAK_ITEMS = 256;

int peak_blocks(long n_cells) {
    const long per = (long)PEAK_THREADS * PEAK_ITEMS;
    long nb = (n_cells + per - 1) / per;
    if (nb < 1)
        nb = 1;
    if (nb > 65535)
        nb = 65535; // grid-stride beyond this
    return (int)nb;
}

size_t peak_scratch_bytes(long n_cells) {
    const int nb = peak_blocks(n_cells);
    return (size_t)nb * (sizeof(float) + sizeof(double) + sizeof(long long));
}

__global__ void peak_partial_kernel(const float* __restrict__ surf, long n_cells,
                                    float* __restrict__ bmax, double* __restrict__ bsum,
                                    long long* __restrict__ bidx) {
    __shared__ float smax[PEAK_THREADS];
    __shared__ double ssum[PEAK_THREADS];
    __shared__ long long sidx[PEAK_THREADS];

    const int t = threadIdx.x;
    float m = -1.0f;
    // Per-thread partial in FLOAT, promoted to double only for the block reduction. Accumulating
    // the grid-stride loop in double costs ~4x here: FP64 runs at 1/64 rate on an L40S and this
    // kernel is otherwise pure bandwidth. A thread sees ~n_cells/(blocks*threads) values of
    // O(100), so a float partial carries ~1e-7*sqrt(n) relative error -- far below what the
    // peak/mean RATIO needs, while the cross-thread and cross-block sums, which is where the
    // 128M-cell cancellation risk actually lives, stay in double.
    float s = 0.0f;
    long long mi = -1;
    for (long i = (long)blockIdx.x * blockDim.x + t; i < n_cells; i += (long)gridDim.x * blockDim.x) {
        const float v = surf[i];
        s += v;
        if (v > m) {
            m = v;
            mi = i;
        }
    }
    smax[t] = m;
    ssum[t] = (double)s;
    sidx[t] = mi;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            ssum[t] += ssum[t + stride];
            // strict >: on a tie keep the LOWER flat index, matching channelized_peak's
            // "first strict maximum".
            if (smax[t + stride] > smax[t]
                || (smax[t + stride] == smax[t] && sidx[t + stride] < sidx[t])) {
                smax[t] = smax[t + stride];
                sidx[t] = sidx[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        bmax[blockIdx.x] = smax[0];
        bsum[blockIdx.x] = ssum[0];
        bidx[blockIdx.x] = sidx[0];
    }
}

__global__ void peak_final_kernel(const float* __restrict__ bmax, const double* __restrict__ bsum,
                                  const long long* __restrict__ bidx, int nb,
                                  float2* __restrict__ out_val, long long* __restrict__ out_idx) {
    float m = -1.0f;
    double s = 0.0;
    long long mi = -1;
    for (int i = 0; i < nb; ++i) {
        s += bsum[i];
        if (bmax[i] > m || (bmax[i] == m && bidx[i] < mi)) {
            m = bmax[i];
            mi = bidx[i];
        }
    }
    out_val[0] = make_float2(m, (float)s);
    out_idx[0] = mi;
    // The sum is also returned in double via out_val.y's float cast being lossy, so callers that
    // need the exact mean read it from the scratch reduction instead. Kept float here because
    // the peak/mean RATIO only needs ~1e-7, and 128M cells of O(100) sum to O(1e10) which a
    // float still represents to ~1e3 -- 1e-7 relative. Good enough for an SNR, deliberately.
}

void launch_peak(const float* surf, long n_cells, float2* out_val, long long* out_idx,
                 void* scratch, cudaStream_t stream) {
    if (n_cells <= 0)
        return;
    const int nb = peak_blocks(n_cells);
    float* bmax = (float*)scratch;
    double* bsum = (double*)(bmax + nb);
    long long* bidx = (long long*)(bsum + nb);
    peak_partial_kernel<<<nb, PEAK_THREADS, 0, stream>>>(surf, n_cells, bmax, bsum, bidx);
    peak_final_kernel<<<1, 1, 0, stream>>>(bmax, bsum, bidx, nb, out_val, out_idx);
}

} // namespace gnss_cuda
