#include "cudaGnssAcquireKernel.hpp"

#include <cuda_runtime.h>
#include <vector>

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
// Folded aggregate over the channel-major layout [nc][n_dop][Mp], cuFFT's natural inverse output.
// Per (d,q) row:
//
//   1. fold nc channels into s_cols bins in shared memory   (nc adds, not nc*s_cols MACs)
//   2. one s_cols-point inverse DFT of the bins              (sign e^{+i}, matching the ramp)
//   3. accumulate |.|^2
//
// s_cols must be a power of two; GnssCudaAcquire refuses other values. The DFT is radix-2
// decimation-in-time: bit-reverse in shared memory, then log2(s_cols) butterfly stages, with
// blockDim == s_cols, so half the threads idle in each stage. Channels congruent mod s_cols share
// a bin, so the fold's add is atomic. W[j * (s_cols/len)] == e^{+2 pi i j / len} makes it the
// inverse transform, matching the ramp's e^{+i...}; with the sign wrong the peak lands at
// s_cols - s, which looks like an indexing bug.
//
// A channel-fastest layout would need cuFFT to write with ostride = nc: 8-byte writes at an
// nc*8 = 632-byte stride, each in its own memory sector. Here cuFFT writes contiguously and the
// transpose happens through a shared-memory tile on the read side: each (channel, q-tile) load
// is Q consecutive complex values.
//
// One block owns one Doppler bin and one tile of Q coarse lags, and walks the tile's rows.
// ---------------------------------------------------------------------------------------------
extern __shared__ float2 smem[]; // [s_cols] fold/FFT buffer, [s_cols/2] twiddles, [nc][Q] tile

__global__ void agg_folded_cdq_kernel(const float2* __restrict__ P, float* __restrict__ surf,
                                      const int* __restrict__ bin,
                                      const float2* __restrict__ twiddle, int nc, int n_dop, int Mp,
                                      int s_cols, int log2n, int Q) {
    const int d = blockIdx.y;
    const int q0 = blockIdx.x * Q;
    if (d >= n_dop || q0 >= Mp)
        return;
    const int t = threadIdx.x;
    const int qn = min(Q, Mp - q0);

    float2* F = smem;                    // [s_cols]
    float2* W = F + s_cols;              // [s_cols/2]
    float2* Ps = W + s_cols / 2;         // [nc][Q]

    for (int i = t; i < s_cols / 2; i += blockDim.x)
        W[i] = twiddle[i];
    // Stage the tile: Q consecutive lags for every channel. Coalesced within a channel.
    for (int i = t; i < nc * qn; i += blockDim.x) {
        const int c = i / qn, j = i - c * qn;
        Ps[(long)c * Q + j] = P[((long)c * n_dop + d) * Mp + q0 + j];
    }
    __syncthreads();

    for (int j = 0; j < qn; ++j) {
        for (int i = t; i < s_cols; i += blockDim.x)
            F[i] = make_float2(0.0f, 0.0f);
        __syncthreads();
        for (int c = t; c < nc; c += blockDim.x) {
            const float2 p = Ps[(long)c * Q + j];
            const int b = bin[c];
            atomicAdd(&F[b].x, p.x);
            atomicAdd(&F[b].y, p.y);
        }
        __syncthreads();

        float2 v = make_float2(0.0f, 0.0f);
        if (t < s_cols)
            v = F[__brev((unsigned)t) >> (32 - log2n)];
        __syncthreads();
        if (t < s_cols)
            F[t] = v;
        __syncthreads();

        for (int len = 2; len <= s_cols; len <<= 1) {
            const int half = len >> 1;
            if (t < s_cols / 2) {
                const int jj = t & (half - 1);
                const int base = (t / half) * len;
                const int i0 = base + jj, i1 = i0 + half;
                const float2 w = W[jj * (s_cols / len)];
                const float2 a = F[i0];
                const float2 b2 = cmul(F[i1], w);
                F[i0] = cadd(a, b2);
                F[i1] = csub(a, b2);
            }
            __syncthreads();
        }
        if (t < s_cols) {
            const float2 dv = F[t];
            surf[((long)d * Mp + q0 + j) * s_cols + t] += dv.x * dv.x + dv.y * dv.y;
        }
    }
}

void launch_aggregate_cdq(const float2* P, float* surf, const int* bin, const float2* twiddle,
                          int nc, int n_dop, int Mp, int s_cols, cudaStream_t stream) {
    if (nc <= 0 || n_dop <= 0 || Mp <= 0 || s_cols <= 0)
        return;
    int log2n = 0;
    while ((1 << log2n) < s_cols)
        ++log2n;
    // Q trades staging shared memory against block count. At Q = 32 the tile is nc*32*8 bytes,
    // 20 KB for CHORD's 79 channels, and the whole allocation stays inside the 48 KB default
    // (no opt-in is requested) for nc up to 186 at s_cols = 128.
    const int Q = 32;
    const size_t shm = (size_t)(s_cols + s_cols / 2 + nc * Q) * sizeof(float2);
    const dim3 grd((unsigned)((Mp + Q - 1) / Q), (unsigned)n_dop);
    agg_folded_cdq_kernel<<<grd, s_cols, shm, stream>>>(P, surf, bin, twiddle, nc, n_dop, Mp,
                                                        s_cols, log2n, Q);
}

// ---------------------------------------------------------------------------------------------
// Peak and sum, on device.
//
// channelized_peak takes the first strict maximum and normalizes by the surface mean, so this
// reproduces both: ties resolve to the lowest flat index, and the block and grid sums are in
// double (128M cells of O(100) summed in float would lose the mean, the denominator of every
// reported SNR, to round-off). A thread's own partial is in float, promoted to double for the
// block reduction: FP64 is 1/64 rate on an L40S and the reduction is otherwise
// bandwidth-bound, and a thread sums few enough values that its float partial is good to
// ~1e-7*sqrt(n) relative.
// ---------------------------------------------------------------------------------------------
static constexpr int PEAK_THREADS = 256;
// Cells streamed per thread before the block reduction, so the eight reduction stages are
// amortized over enough streaming; at CHORD's blind dimensions that is still 7 blocks per
// Doppler bin, ~2250 in all.
static constexpr int PEAK_ITEMS = 256;

// ---------------------------------------------------------------------------------------------
// Correlate front end.
// ---------------------------------------------------------------------------------------------

__global__ void gather_pad_kernel(const float2* __restrict__ snap, const int* __restrict__ cov,
                                  float2* __restrict__ out, int n_chan_total, int M, int Mp,
                                  int nc) {
    const int c = blockIdx.y;
    if (c >= nc)
        return;
    const int lc = cov[c];
    for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < Mp; m += gridDim.x * blockDim.x)
        out[(long)c * Mp + m] =
            (m < M) ? snap[(long)m * n_chan_total + lc] : make_float2(0.0f, 0.0f);
}

void launch_gather_pad(const float2* snap, const int* cov, float2* out, int n_chan_total, int M,
                       int Mp, int nc, cudaStream_t stream) {
    if (nc <= 0 || Mp <= 0)
        return;
    const dim3 blk(256), grd((unsigned)((Mp + 255) / 256), (unsigned)nc);
    gather_pad_kernel<<<grd, blk, 0, stream>>>(snap, cov, out, n_chan_total, M, Mp, nc);
}

__global__ void materialise_kernel(const float2* __restrict__ head, const float2* __restrict__ tail,
                                   const float* __restrict__ sgn0, const float* __restrict__ sgn1,
                                   float2* __restrict__ R, int nc, int Mp) {
    const int c = blockIdx.y;
    if (c >= nc)
        return;
    for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < Mp; m += gridDim.x * blockDim.x) {
        const long i = (long)c * Mp + m;
        const float2 h = head[i], t = tail[i];
        const float a0 = sgn0[m], a1 = sgn1[m];
        R[i] = make_float2(h.x * a0 + t.x * a1, h.y * a0 + t.y * a1);
    }
}

void launch_materialise(const float2* head, const float2* tail, const float* sgn0,
                        const float* sgn1, float2* R, int nc, int Mp, cudaStream_t stream) {
    if (nc <= 0 || Mp <= 0)
        return;
    const dim3 blk(256), grd((unsigned)((Mp + 255) / 256), (unsigned)nc);
    materialise_kernel<<<grd, blk, 0, stream>>>(head, tail, sgn0, sgn1, R, nc, Mp);
}

__global__ void dopmul_kernel(const float2* __restrict__ A, const float2* __restrict__ B,
                              const int* __restrict__ bshift, float2* __restrict__ X, int nc,
                              int n_dop, int Mp, float inv_mp) {
    const int c = blockIdx.z;
    const int d = blockIdx.y;
    if (c >= nc || d >= n_dop)
        return;
    const int b = bshift[d];
    const float2* __restrict__ Ac = A + (long)c * Mp;
    const float2* __restrict__ Bc = B + (long)c * Mp;
    float2* __restrict__ Xc = X + ((long)c * n_dop + d) * Mp;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < Mp; k += gridDim.x * blockDim.x) {
        int j = k + b;
        if (j >= Mp)
            j -= Mp; // b is already reduced mod Mp, so one conditional suffices
        const float2 a = Ac[j];
        const float2 bb = Bc[k];
        // a * conj(bb), scaled by cuFFT's missing 1/Mp
        Xc[k] = make_float2((a.x * bb.x + a.y * bb.y) * inv_mp,
                            (a.y * bb.x - a.x * bb.y) * inv_mp);
    }
}

void launch_dopmul(const float2* A, const float2* B, const int* bshift, float2* X, int nc,
                   int n_dop, int Mp, cudaStream_t stream) {
    if (nc <= 0 || n_dop <= 0 || Mp <= 0)
        return;
    const dim3 blk(256), grd((unsigned)((Mp + 255) / 256), (unsigned)n_dop, (unsigned)nc);
    dopmul_kernel<<<grd, blk, 0, stream>>>(A, B, bshift, X, nc, n_dop, Mp, 1.0f / (float)Mp);
}

// Per-Doppler reduction. The search needs the max over lag of every Doppler slice, not just the
// global one: the pair-sum plane selection and peak_from_reduction's sub-grid Doppler refine and
// peak_interp all read it.
//
// One block row per Doppler bin. Slices are contiguous (the surface is [d][Mp][s_cols]), so this
// streams like the global reduction, with the grid's y axis selecting the output.
__global__ void dop_partial_kernel(const float* __restrict__ surf, long cells_per_dop,
                                   float* __restrict__ bmax, double* __restrict__ bsum,
                                   long long* __restrict__ bidx, int nblk) {
    __shared__ float smax[PEAK_THREADS];
    __shared__ double ssum[PEAK_THREADS];
    __shared__ long long sidx[PEAK_THREADS];

    const int d = blockIdx.y;
    const float* base = surf + (long)d * cells_per_dop;
    const int t = threadIdx.x;
    float m = -1.0f, s = 0.0f;
    long long mi = -1;
    for (long i = (long)blockIdx.x * blockDim.x + t; i < cells_per_dop;
         i += (long)gridDim.x * blockDim.x) {
        const float v = base[i];
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
            if (smax[t + stride] > smax[t]
                || (smax[t + stride] == smax[t] && sidx[t + stride] < sidx[t])) {
                smax[t] = smax[t + stride];
                sidx[t] = sidx[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        const long o = (long)d * nblk + blockIdx.x;
        bmax[o] = smax[0];
        bsum[o] = ssum[0];
        bidx[o] = sidx[0];
    }
}

int dop_blocks(long cells_per_dop) {
    const long per = (long)PEAK_THREADS * PEAK_ITEMS;
    long nb = (cells_per_dop + per - 1) / per;
    return (int)(nb < 1 ? 1 : (nb > 4096 ? 4096 : nb));
}

size_t peak_dop_scratch_bytes(long cells_per_dop, int n_dop) {
    const long nb = (long)dop_blocks(cells_per_dop) * n_dop;
    return (size_t)nb * (sizeof(double) + sizeof(long long) + sizeof(float));
}

void launch_peak_dop(const float* surf, int n_dop, long cells_per_dop, float* h_max, double* h_sum,
                     long long* h_idx, void* scratch, cudaStream_t stream) {
    if (n_dop <= 0 || cells_per_dop <= 0)
        return;
    const int nblk = dop_blocks(cells_per_dop);
    const long tot = (long)nblk * n_dop;
    // Partials are laid out widest first: a float array at the front would leave the double
    // array only 4-byte aligned whenever its length is odd, and the kernel would fault on a
    // misaligned address.
    double* bsum = (double*)scratch;
    long long* bidx = (long long*)(bsum + tot);
    float* bmax = (float*)(bidx + tot);

    const dim3 grd((unsigned)nblk, (unsigned)n_dop);
    dop_partial_kernel<<<grd, PEAK_THREADS, 0, stream>>>(surf, cells_per_dop, bmax, bsum, bidx,
                                                         nblk);
    // Final combine on the host: n_dop*nblk is a few thousand values, so a second kernel and a
    // second synchronization buy nothing, and the caller needs the per-Doppler arrays on the
    // host anyway.
    std::vector<float> hm((size_t)tot);
    std::vector<double> hs((size_t)tot);
    std::vector<long long> hi((size_t)tot);
    cudaMemcpyAsync(hm.data(), bmax, hm.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(hs.data(), bsum, hs.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(hi.data(), bidx, hi.size() * sizeof(long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (int d = 0; d < n_dop; ++d) {
        float m = -1.0f;
        double s = 0.0;
        long long mi = -1;
        for (int b = 0; b < nblk; ++b) {
            const long o = (long)d * nblk + b;
            s += hs[(size_t)o];
            if (hm[(size_t)o] > m || (hm[(size_t)o] == m && hi[(size_t)o] < mi)) {
                m = hm[(size_t)o];
                mi = hi[(size_t)o];
            }
        }
        h_max[d] = m;
        h_sum[d] = s;
        h_idx[d] = mi;
    }
}

} // namespace gnss_cuda
