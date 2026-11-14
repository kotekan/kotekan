#include "fEngineKernels.hpp"

__global__ static void unpack12_f32(const uint64_t* __restrict__ lut, int num_packets, int pairs,
                                    float2* __restrict__ out) {
    for (int p = blockIdx.x; p < num_packets; p += gridDim.x) {
        const uint8_t* __restrict__ slot = (const uint8_t*)(uintptr_t)lut[p];
        float2* __restrict__ o = out + (size_t)p * pairs;
        for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
            const unsigned b0 = slot[3 * i], b1 = slot[3 * i + 1], b2 = slot[3 * i + 2];
            const int s0 = (int)(b0 | ((b1 & 0xf) << 8)) - 2048;
            const int s1 = (int)((b1 >> 4) | (b2 << 4)) - 2048;
            o[i] = make_float2(s0 * (1.0f / 2048.0f), s1 * (1.0f / 2048.0f));
        }
    }
}

void gpu_unpack12(const uint64_t* lut, int num_packets, int samples_per_packet, float* out,
                  cudaStream_t stream) {
    const int blocks = num_packets < 4096 ? num_packets : 4096;
    unpack12_f32<<<blocks, 256, 0, stream>>>(lut, num_packets, samples_per_packet / 2,
                                             (float2*)out);
}

__global__ static void untangle_quantize(const float2* __restrict__ spec, size_t m,
                                         const float2* __restrict__ w_hi,
                                         const float2* __restrict__ w_lo, float scale,
                                         uint8_t* __restrict__ out) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < m;
         k += (size_t)gridDim.x * blockDim.x) {
        const float2 zk = spec[k];
        const float2 zmk = spec[k == 0 ? 0 : m - k];
        // Ze = (Z[k] + conj(Z[m-k])) / 2, Zo = (Z[k] - conj(Z[m-k])) / 2i.
        const float er = 0.5f * (zk.x + zmk.x);
        const float ei = 0.5f * (zk.y - zmk.y);
        const float our = 0.5f * (zk.y + zmk.y);
        const float oui = 0.5f * (zmk.x - zk.x);
        const float2 wh = w_hi[k >> 12];
        const float2 wl = w_lo[k & 4095];
        const float2 w = make_float2(wh.x * wl.x - wh.y * wl.y, wh.x * wl.y + wh.y * wl.x);
        const float xr = er + w.x * our - w.y * oui;
        const float xi = ei + w.x * oui + w.y * our;
        const int re = max(-7, min(7, __float2int_rn(scale * xr)));
        const int im = max(-7, min(7, __float2int_rn(scale * xi)));
        out[k] = (uint8_t)(((re + 8) << 4) | (im + 8));
    }
}

void gpu_untangle_quantize(const float2* spec, size_t m, const float2* w_hi, const float2* w_lo,
                           float scale, uint8_t* out, cudaStream_t stream) {
    const size_t want = (m + 255) / 256;
    const int blocks = want < 4096 ? (int)want : 4096;
    untangle_quantize<<<blocks, 256, 0, stream>>>(spec, m, w_hi, w_lo, scale, out);
}

// ---- fp16 four-step kernels. All three work in 32x32 tiles with (32, 8) threads: smem
// staging turns the transposed access of one side into coalesced rows on both sides.

__global__ static void unpack12_fp16_t(const uint64_t* __restrict__ lut, int sp, size_t m1,
                                       size_t m2, __half2* __restrict__ out) {
    __shared__ __half2 tile[32][33];
    const size_t a0 = blockIdx.x * 32; // index along m1
    const size_t b0 = blockIdx.y * 32; // index along m2
    for (int l = threadIdx.y; l < 32; l += 8) {
        // One row = 32 complex = 64 contiguous real samples starting at an even multiple
        // of 64, so it never straddles a packet (sp is a multiple of 64).
        const size_t s0 = 2 * ((a0 + l) * m2 + b0);
        const uint8_t* __restrict__ base =
            (const uint8_t*)(uintptr_t)lut[s0 / sp] + 3 * ((s0 % sp) / 2);
        const int c = threadIdx.x;
        const unsigned b_0 = base[3 * c], b_1 = base[3 * c + 1], b_2 = base[3 * c + 2];
        const int v0 = (int)(b_0 | ((b_1 & 0xf) << 8)) - 2048;
        const int v1 = (int)((b_1 >> 4) | (b_2 << 4)) - 2048;
        tile[l][c] = __floats2half2_rn(v0 * (1.0f / 2048.0f), v1 * (1.0f / 2048.0f));
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8)
        out[(b0 + r) * m1 + a0 + threadIdx.x] = tile[threadIdx.x][r];
}

void gpu_unpack12_fp16_t(const uint64_t* lut, int samples_per_packet, size_t m1, size_t m2,
                         __half2* out, cudaStream_t stream) {
    dim3 grid(m1 / 32, m2 / 32), block(32, 8);
    unpack12_fp16_t<<<grid, block, 0, stream>>>(lut, samples_per_packet, m1, m2, out);
}

__device__ static float2 twiddle_hilo(const float2* __restrict__ w_hi,
                                      const float2* __restrict__ w_lo, uint64_t e) {
    const float2 wh = w_hi[e >> 12];
    const float2 wl = w_lo[e & 4095];
    return make_float2(wh.x * wl.x - wh.y * wl.y, wh.x * wl.y + wh.y * wl.x);
}

__global__ static void twiddle_transpose(const __half2* __restrict__ in, size_t m1, size_t m2,
                                         const float2* __restrict__ w_hi,
                                         const float2* __restrict__ w_lo, float inv_m1,
                                         __half2* __restrict__ out) {
    __shared__ __half2 tile[32][33];
    const size_t k0 = blockIdx.x * 32; // index along m1 (k1)
    const size_t b0 = blockIdx.y * 32; // index along m2 (b)
    for (int l = threadIdx.y; l < 32; l += 8)
        tile[l][threadIdx.x] = in[(b0 + l) * m1 + k0 + threadIdx.x];
    __syncthreads();
    const uint64_t mask = m1 * m2 - 1; // m is a power of two
    for (int r = threadIdx.y; r < 32; r += 8) {
        const size_t k1 = k0 + r;
        const size_t b = b0 + threadIdx.x;
        const float2 v = __half22float2(tile[threadIdx.x][r]);
        const float2 w = twiddle_hilo(w_hi, w_lo, 2 * ((k1 * b) & mask));
        out[k1 * m2 + b] =
            __floats2half2_rn(inv_m1 * (v.x * w.x - v.y * w.y), inv_m1 * (v.x * w.y + v.y * w.x));
    }
}

void gpu_twiddle_transpose(const __half2* in, size_t m1, size_t m2, const float2* w_hi,
                           const float2* w_lo, float inv_m1, __half2* out, cudaStream_t stream) {
    dim3 grid(m1 / 32, m2 / 32), block(32, 8);
    twiddle_transpose<<<grid, block, 0, stream>>>(in, m1, m2, w_hi, w_lo, inv_m1, out);
}

__global__ static void quantize_fourstep(const __half2* __restrict__ spec, size_t m1, size_t m2,
                                         const float2* __restrict__ w_hi,
                                         const float2* __restrict__ w_lo, float scale,
                                         uint8_t* __restrict__ out) {
    __shared__ __half2 fwd[32][33];
    __shared__ __half2 par[32][33];
    const size_t k10 = blockIdx.x * 32;
    const size_t k20 = blockIdx.y * 32;
    for (int l = threadIdx.y; l < 32; l += 8) {
        const size_t k1 = k10 + l;
        const int c = threadIdx.x;
        fwd[l][c] = spec[k1 * m2 + k20 + c];
        // The conjugate partner of (k1, k2) is (m1 - k1, m2 - 1 - k2), except row 0 pairs
        // with itself at column (m2 - k2) mod m2.
        if (k1 == 0)
            par[l][c] = spec[(m2 - (k20 + c)) % m2];
        else
            par[l][c] = spec[(m1 - k1) * m2 + m2 - 1 - (k20 + c)];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const size_t k1 = k10 + threadIdx.x;
        const size_t k2 = k20 + r;
        const size_t k = k2 * m1 + k1;
        const float2 zk = __half22float2(fwd[threadIdx.x][r]);
        const float2 zmk = __half22float2(par[threadIdx.x][r]);
        const float er = 0.5f * (zk.x + zmk.x);
        const float ei = 0.5f * (zk.y - zmk.y);
        const float our = 0.5f * (zk.y + zmk.y);
        const float oui = 0.5f * (zmk.x - zk.x);
        const float2 w = twiddle_hilo(w_hi, w_lo, k);
        const float xr = er + w.x * our - w.y * oui;
        const float xi = ei + w.x * oui + w.y * our;
        const int re = max(-7, min(7, __float2int_rn(scale * xr)));
        const int im = max(-7, min(7, __float2int_rn(scale * xi)));
        out[k] = (uint8_t)(((re + 8) << 4) | (im + 8));
    }
}

void gpu_quantize_fourstep(const __half2* spec, size_t m1, size_t m2, const float2* w_hi,
                           const float2* w_lo, float scale, uint8_t* out, cudaStream_t stream) {
    dim3 grid(m1 / 32, m2 / 32), block(32, 8);
    quantize_fourstep<<<grid, block, 0, stream>>>(spec, m1, m2, w_hi, w_lo, scale, out);
}
