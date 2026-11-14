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
