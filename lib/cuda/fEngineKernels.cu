#include "fEngineKernels.hpp"
#include "tensorFFT.hpp" // for tensor_fft_bin, tensor_fft_pos

#include <cstring> // for memcpy

__global__ static void unpack12_f32(const uint64_t* __restrict__ lut, int num_packets, int pairs,
                                    float2* __restrict__ out) {
    for (int p = blockIdx.x; p < num_packets; p += gridDim.x) {
        const uint8_t* __restrict__ slot = (const uint8_t*)(uintptr_t)lut[p];
        float2* __restrict__ o = out + (size_t)p * pairs;
        for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
            // Two 12-bit two's complement samples, sign-extended.
            const unsigned b0 = slot[3 * i], b1 = slot[3 * i + 1], b2 = slot[3 * i + 2];
            const int s0 = (int)((b0 | ((b1 & 0xf) << 8)) ^ 0x800) - 0x800;
            const int s1 = (int)(((b1 >> 4) | (b2 << 4)) ^ 0x800) - 0x800;
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

// The 4+4 byte of g * (vr, vi): round to nearest even, clamp +-7, offset by 8, real high.
__device__ static uint8_t encode(float vr, float vi, float g) {
    const int re = max(-7, min(7, __float2int_rn(g * vr)));
    const int im = max(-7, min(7, __float2int_rn(g * vi)));
    return (uint8_t)(((re + 8) << 4) | (im + 8));
}

__device__ static float2 twiddle_hilo(const float2* __restrict__ w_hi,
                                      const float2* __restrict__ w_lo, uint64_t e) {
    const float2 wh = w_hi[e >> 12];
    const float2 wl = w_lo[e & 4095];
    return make_float2(wh.x * wl.x - wh.y * wl.y, wh.x * wl.y + wh.y * wl.x);
}

// scale * X[k] from zk = Z[k], zmk = Z[m - k] and w = W^k: X[k] = Ze + W^k Zo with
// Ze = (Z[k] + conj(Z[m-k])) / 2, Zo = (Z[k] - conj(Z[m-k])) / 2i.
__device__ static float2 untangle(float2 zk, float2 zmk, float2 w, float scale) {
    const float er = 0.5f * (zk.x + zmk.x);
    const float ei = 0.5f * (zk.y - zmk.y);
    const float our = 0.5f * (zk.y + zmk.y);
    const float oui = 0.5f * (zmk.x - zk.x);
    return make_float2(scale * (er + w.x * our - w.y * oui), scale * (ei + w.x * oui + w.y * our));
}

// One bin per thread. Block 2j quantizes the 256 bins of chunk j and block 2j + 1 those of the
// j-th chunk from the end, which holds all but one of their conjugate partners m - k, so the
// partner loads of each block are its neighbour's own loads and come from L2: the spectrum
// crosses DRAM about once instead of twice.
__global__ static void untangle_quantize(const float2* __restrict__ spec, size_t m,
                                         const float2* __restrict__ w_hi,
                                         const float2* __restrict__ w_lo, float scale, float gain,
                                         const float* __restrict__ gains, float* __restrict__ stats,
                                         uint8_t* __restrict__ out) {
    const size_t chunks = (m + 255) / 256;
    const size_t c = blockIdx.x % 2 ? chunks - 1 - blockIdx.x / 2 : blockIdx.x / 2;
    const size_t k = c * 256 + threadIdx.x;
    if (k >= m)
        return;
    const float2 x =
        untangle(spec[k], spec[k == 0 ? 0 : m - k], twiddle_hilo(w_hi, w_lo, k), scale);
    if (stats)
        stats[k] += x.x * x.x + x.y * x.y;
    out[k] = encode(x.x, x.y, gains ? gain * gains[k] : gain);
}

void gpu_untangle_quantize(const float2* spec, size_t m, const float2* w_hi, const float2* w_lo,
                           float scale, float gain, const float* gains, float* stats, uint8_t* out,
                           cudaStream_t stream) {
    untangle_quantize<<<(unsigned)((m + 255) / 256), 256, 0, stream>>>(spec, m, w_hi, w_lo, scale,
                                                                       gain, gains, stats, out);
}

// ---- Four-step kernels, templated on the complex storage type (__half2 for the fp16
// path, float2 for the TF32 tensor path). The unpack and twiddle passes work in 32x32 tiles
// with (32, 8) threads, the quantizer in taller tiles: smem staging turns the transposed
// access of one side into coalesced rows on both sides.

__device__ static inline float2 tofloat2(const __half2 v) {
    return __half22float2(v);
}
__device__ static inline float2 tofloat2(const float2 v) {
    return v;
}
__device__ static inline void store2(__half2& d, float a, float b) {
    d = __floats2half2_rn(a, b);
}
__device__ static inline void store2(float2& d, float a, float b) {
    d = make_float2(a, b);
}

// Blocks are numbered with b fastest, so the blocks resident at once read neighbouring 96-byte
// pieces of the same 32 rows. Numbered with a fastest they would read every row of the window
// at once, which over host payload slots loses most of the read bandwidth once a window passes
// about 0.4 GiB (2^28 points) on GB10.
template<typename T2>
__global__ static void unpack12_t(const uint64_t* __restrict__ lut, int sp, size_t m1, size_t m2,
                                  T2* __restrict__ out) {
    __shared__ T2 tile[32][33];
    const size_t a0 = blockIdx.y * 32; // index along m1
    const size_t b0 = blockIdx.x * 32; // index along m2
    for (int l = threadIdx.y; l < 32; l += 8) {
        // One row = 32 complex = 64 contiguous real samples starting at an even multiple
        // of 64, so it never straddles a packet (sp is a multiple of 64).
        const size_t s0 = 2 * ((a0 + l) * m2 + b0);
        const uint8_t* __restrict__ base =
            (const uint8_t*)(uintptr_t)lut[s0 / sp] + 3 * ((s0 % sp) / 2);
        const int c = threadIdx.x;
        const unsigned b_0 = base[3 * c], b_1 = base[3 * c + 1], b_2 = base[3 * c + 2];
        const int v0 = (int)((b_0 | ((b_1 & 0xf) << 8)) ^ 0x800) - 0x800;
        const int v1 = (int)(((b_1 >> 4) | (b_2 << 4)) ^ 0x800) - 0x800;
        store2(tile[l][c], v0 * (1.0f / 2048.0f), v1 * (1.0f / 2048.0f));
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8)
        out[(b0 + r) * m1 + a0 + threadIdx.x] = tile[threadIdx.x][r];
}

void gpu_unpack12_fp16_t(const uint64_t* lut, int samples_per_packet, size_t m1, size_t m2,
                         __half2* out, cudaStream_t stream) {
    dim3 grid(m2 / 32, m1 / 32), block(32, 8);
    unpack12_t<<<grid, block, 0, stream>>>(lut, samples_per_packet, m1, m2, out);
}

void gpu_unpack12_f32_t(const uint64_t* lut, int samples_per_packet, size_t m1, size_t m2,
                        float2* out, cudaStream_t stream) {
    dim3 grid(m2 / 32, m1 / 32), block(32, 8);
    unpack12_t<<<grid, block, 0, stream>>>(lut, samples_per_packet, m1, m2, out);
}

// split1 is the first FFT's tensorFFT.hpp split: in holds bin k1 of each row at position
// tensor_fft_pos(k1, split1). A tile covers 32 positions, so its rows still load
// coalesced, and each position's bin picks the output row.
template<typename T2>
__global__ static void
twiddle_transpose(const T2* __restrict__ in, size_t m1, size_t m2, const float2* __restrict__ w_hi,
                  const float2* __restrict__ w_lo, float inv_m1, int split1, T2* __restrict__ out) {
    __shared__ T2 tile[32][33];
    const size_t p0 = blockIdx.x * 32; // position along m1
    const size_t b0 = blockIdx.y * 32; // index along m2 (b)
    for (int l = threadIdx.y; l < 32; l += 8)
        tile[l][threadIdx.x] = in[(b0 + l) * m1 + p0 + threadIdx.x];
    __syncthreads();
    const uint64_t mask = m1 * m2 - 1; // m is a power of two
    for (int r = threadIdx.y; r < 32; r += 8) {
        const size_t k1 = tensor_fft_bin(p0 + r, split1);
        const size_t b = b0 + threadIdx.x;
        const float2 v = tofloat2(tile[threadIdx.x][r]);
        const float2 w = twiddle_hilo(w_hi, w_lo, 2 * ((k1 * b) & mask));
        store2(out[k1 * m2 + b], inv_m1 * (v.x * w.x - v.y * w.y),
               inv_m1 * (v.x * w.y + v.y * w.x));
    }
}

void gpu_twiddle_transpose(const __half2* in, size_t m1, size_t m2, const float2* w_hi,
                           const float2* w_lo, float inv_m1, int split1, __half2* out,
                           cudaStream_t stream) {
    dim3 grid(m1 / 32, m2 / 32), block(32, 8);
    twiddle_transpose<<<grid, block, 0, stream>>>(in, m1, m2, w_hi, w_lo, inv_m1, split1, out);
}

void gpu_twiddle_transpose_f32(const float2* in, size_t m1, size_t m2, const float2* w_hi,
                               const float2* w_lo, float inv_m1, float2* out, cudaStream_t stream) {
    dim3 grid(m1 / 32, m2 / 32), block(32, 8);
    twiddle_transpose<<<grid, block, 0, stream>>>(in, m1, m2, w_hi, w_lo, inv_m1, 0, out);
}

// Columns of quantizer tiles per band (see quantize_fourstep).
static constexpr size_t QUANTIZE_BAND = 32;

// K adjacent floats as one access, and K quantized bytes as one store.
template<int K>
struct quantize_lanes;
template<>
struct quantize_lanes<2> {
    using floats = float2;
    using bytes = uint16_t;
};
template<>
struct quantize_lanes<1> {
    using floats = float;
    using bytes = uint8_t;
};

// Tiles of 32K rows (k1) by 32 columns (k2): spectrum rows load coalesced into shared memory,
// then each lane quantizes K adjacent bins k1 of one k2, so a warp stores 32K contiguous
// output bytes. Tile element (l, c) sits at l * 32 + (c ^ l / K): the row stores and the
// lanes' K-row reads both hit 32 distinct banks. Tile t and tile T - 1 - t, which holds all
// but one row of t's conjugate partners, run as neighbouring blocks, so the partner loads of
// each are the other's own loads and come from L2: the spectrum crosses DRAM about once
// instead of twice.
//
// Tiles are numbered in bands of QUANTIZE_BAND tile columns, k1 fastest within a band, so the
// blocks resident at once write neighbouring pieces of a few hundred output rows. With k2
// fastest throughout, their stores spread over the whole spectrum, which over a host output
// frame loses most of the write bandwidth past about 2^28 points; with k1 fastest throughout,
// the spectrum reads lose their locality instead.
//
// split2 is the second FFT's tensorFFT.hpp split. A tile's 32 columns are positions, with
// k2 = tensor_fft_bin(position, split2); the partner column m2 - 1 - k2 sits at position
// m2 - 1 - position, so only row 0 and the output rows need the mapping.
template<typename T2, int K>
__global__ static void __launch_bounds__(256)
    quantize_fourstep(const T2* __restrict__ spec, size_t m1, size_t m2,
                      const float2* __restrict__ w_hi, const float2* __restrict__ w_lo, float scale,
                      float gain, const float* __restrict__ gains, float* __restrict__ stats,
                      int split2, uint8_t* __restrict__ out) {
    using floats = typename quantize_lanes<K>::floats;
    using bytes = typename quantize_lanes<K>::bytes;
    __shared__ T2 fwd[32 * K * 32];
    __shared__ T2 par[32 * K * 32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const size_t na = m1 / (32 * K), nb = m2 / 32, tiles = na * nb;
    const size_t t = blockIdx.x % 2 ? tiles - 1 - blockIdx.x / 2 : blockIdx.x / 2;
    const size_t g = nb < QUANTIZE_BAND ? nb : QUANTIZE_BAND, band = t / (g * na);
    const size_t k10 = t % (g * na) / g * 32 * K, k20 = (band * g + t % g) * 32;
    for (int l = warp; l < 32 * K; l += 8) {
        const size_t k1 = k10 + l, p2 = k20 + lane;
        const int s = l * 32 + (lane ^ (l / K));
        fwd[s] = spec[k1 * m2 + p2];
        // The conjugate partner of (k1, k2) is (m1 - k1, m2 - 1 - k2), except row 0 pairs
        // with itself at column (m2 - k2) mod m2.
        par[s] = k1 == 0 ? spec[tensor_fft_pos((m2 - tensor_fft_bin(p2, split2)) % m2, split2)]
                         : spec[(m1 - k1) * m2 + m2 - 1 - p2];
    }
    __syncthreads();
    for (int r = warp; r < 32; r += 8) {
        // The lane's first bin.
        const size_t k = tensor_fft_bin(k20 + r, split2) * m1 + k10 + K * lane;
        float g[K], st[K];
        if (gains) {
            const floats v = *(const floats*)(gains + k);
            memcpy(g, &v, sizeof(v));
        }
        if (stats) {
            const floats v = *(const floats*)(stats + k);
            memcpy(st, &v, sizeof(v));
        }
        uint32_t q = 0;
#pragma unroll
        for (int i = 0; i < K; i++) {
            const int s = (K * lane + i) * 32 + (r ^ lane);
            const float2 x = untangle(tofloat2(fwd[s]), tofloat2(par[s]),
                                      twiddle_hilo(w_hi, w_lo, k + i), scale);
            if (stats)
                st[i] += x.x * x.x + x.y * x.y;
            q |= (uint32_t)encode(x.x, x.y, gains ? gain * g[i] : gain) << (8 * i);
        }
        if (stats) {
            floats v;
            memcpy(&v, st, sizeof(v));
            *(floats*)(stats + k) = v;
        }
        *(bytes*)(out + k) = (bytes)q;
    }
}

// 64-row tiles where m1 allows; 128-row tiles measured slower on GB10 (fewer blocks per SM).
template<typename T2>
static void launch_quantize_fourstep(const T2* spec, size_t m1, size_t m2, const float2* w_hi,
                                     const float2* w_lo, float scale, float gain,
                                     const float* gains, float* stats, int split2, uint8_t* out,
                                     cudaStream_t stream) {
    if (m1 >= 64)
        quantize_fourstep<T2, 2><<<(unsigned)(m1 / 64 * (m2 / 32)), 256, 0, stream>>>(
            spec, m1, m2, w_hi, w_lo, scale, gain, gains, stats, split2, out);
    else
        quantize_fourstep<T2, 1><<<(unsigned)(m1 / 32 * (m2 / 32)), 256, 0, stream>>>(
            spec, m1, m2, w_hi, w_lo, scale, gain, gains, stats, split2, out);
}

void gpu_quantize_fourstep(const __half2* spec, size_t m1, size_t m2, const float2* w_hi,
                           const float2* w_lo, float scale, float gain, const float* gains,
                           float* stats, int split2, uint8_t* out, cudaStream_t stream) {
    launch_quantize_fourstep(spec, m1, m2, w_hi, w_lo, scale, gain, gains, stats, split2, out,
                             stream);
}

void gpu_quantize_fourstep_f32(const float2* spec, size_t m1, size_t m2, const float2* w_hi,
                               const float2* w_lo, float scale, float gain, const float* gains,
                               float* stats, uint8_t* out, cudaStream_t stream) {
    launch_quantize_fourstep(spec, m1, m2, w_hi, w_lo, scale, gain, gains, stats, 0, out, stream);
}
