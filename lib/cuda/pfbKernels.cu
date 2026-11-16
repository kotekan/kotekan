#include "cudaUtils.hpp" // for CHECK_CUDA_ERROR_NON_OO
#include "pfbKernels.hpp"
#include "pfbPrototype.hpp" // for pfb_prototype

#include <cstring> // for memcpy
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

const __half* pfb_proto_shared(int gpu_id, size_t fft_length, int num_taps, bool& reused) {
    static std::mutex lock;
    static std::map<std::tuple<int, size_t, int>, __half*> tables;
    std::lock_guard<std::mutex> hold(lock);
    __half*& table = tables[std::make_tuple(gpu_id, fft_length, num_taps)];
    reused = table != nullptr;
    if (!reused) {
        // The windowed-sinc prototype is symmetric, w[k] == w[L-1-k] exactly in float, so
        // only the first half is stored; the fold kernel mirror-reads the upper taps.
        const std::vector<float> proto = pfb_prototype((int)fft_length, num_taps);
        std::vector<__half> h(proto.size() / 2);
        for (size_t i = 0; i < h.size(); i++)
            h[i] = __float2half_rn(proto[i]);
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&table, h.size() * sizeof(__half)));
        CHECK_CUDA_ERROR_NON_OO(
            cudaMemcpy(table, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice));
    }
    return table;
}

__global__ static void pfb_fold_f32(const float* __restrict__ in0, const float* __restrict__ in1,
                                    const float* __restrict__ in2, const float* __restrict__ in3,
                                    const __half* __restrict__ proto, size_t n,
                                    float* __restrict__ out) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = __half2float(proto[i]) * in0[i] + __half2float(proto[n + i]) * in1[i]
                 + __half2float(proto[2 * n + i]) * in2[i]
                 + __half2float(proto[3 * n + i]) * in3[i];
}

void gpu_pfb_fold(const float* in0, const float* in1, const float* in2, const float* in3,
                  const __half* proto, size_t n_samples, float* out, cudaStream_t stream) {
    const size_t want = (n_samples + 255) / 256;
    const int blocks = want < 4096 ? (int)want : 4096;
    pfb_fold_f32<<<blocks, 256, 0, stream>>>(in0, in1, in2, in3, proto, n_samples, out);
}

// 2K packed 12-bit samples (3K bytes, LSB-first pairs as pack12) from p, read as 32-, 16- or
// 8-bit words for K = 4, 2, 1.
template<int K>
__device__ __forceinline__ void pfb_load12(const uint8_t* __restrict__ p, int (&s)[2 * K]);

template<>
__device__ __forceinline__ void pfb_load12<4>(const uint8_t* __restrict__ p, int (&s)[8]) {
    const uint32_t* __restrict__ w = (const uint32_t*)p;
    const uint32_t w0 = w[0], w1 = w[1], w2 = w[2];
    s[0] = w0 & 0xfff;
    s[1] = (w0 >> 12) & 0xfff;
    s[2] = (w0 >> 24) | ((w1 & 0xf) << 8);
    s[3] = (w1 >> 4) & 0xfff;
    s[4] = (w1 >> 16) & 0xfff;
    s[5] = (w1 >> 28) | ((w2 & 0xff) << 4);
    s[6] = (w2 >> 8) & 0xfff;
    s[7] = w2 >> 20;
}

template<>
__device__ __forceinline__ void pfb_load12<2>(const uint8_t* __restrict__ p, int (&s)[4]) {
    const uint16_t* __restrict__ h = (const uint16_t*)p;
    const uint32_t h0 = h[0], h1 = h[1], h2 = h[2];
    s[0] = h0 & 0xfff;
    s[1] = (h0 >> 12) | ((h1 & 0xff) << 4);
    s[2] = (h1 >> 8) | ((h2 & 0xf) << 8);
    s[3] = h2 >> 4;
}

template<>
__device__ __forceinline__ void pfb_load12<1>(const uint8_t* __restrict__ p, int (&s)[2]) {
    const uint32_t b0 = p[0], b1 = p[1], b2 = p[2];
    s[0] = b0 | ((b1 & 0xf) << 8);
    s[1] = (b1 >> 4) | (b2 << 4);
}

// K adjacent __half2 weights from p as one 16-, 8- or 4-byte load.
template<int K>
struct pfb_words;
template<>
struct pfb_words<4> {
    using type = uint4;
};
template<>
struct pfb_words<2> {
    using type = uint2;
};
template<>
struct pfb_words<1> {
    using type = uint32_t;
};

template<int K>
__device__ __forceinline__ void pfb_load_weights(const __half2* __restrict__ p, __half2 (&w)[K]) {
    const typename pfb_words<K>::type v = *(const typename pfb_words<K>::type*)p;
    memcpy(w, &v, sizeof(v));
}

// Element (a, b) of a 32 x 32K tile, b swizzled so that both the row writes (K adjacent
// columns per lane, straight or mirrored) and the column reads of the transposed store hit
// 32 distinct banks.
template<int K>
__device__ __forceinline__ int pfb_tile_index(int a, int b) {
    return a * 32 * K + ((b ^ ((b >> 5) & (K - 1))) ^ a);
}

// Gather + unpack + fold into the transposed fp16 layout of unpack12_fp16_t
// (fEngineKernels.cu), the fold over the four windows accumulated in fp32 before the one
// rounding to fp16. Only taps 0 and 1 of the symmetric prototype are stored: tap 2 reads tap
// 1 mirrored and tap 3 tap 0 mirrored, each pair swapped. Complex sample j and its mirror
// m1 * m2 - 1 - j use the same four weights with the taps reversed, so each block folds a
// 32 x 32K tile and its mirror tile from one set of weight loads.
//
// A warp reads one tile row per tap, 96K contiguous bytes of one packet: a row starts at a
// multiple of 64K real samples and samples_per_packet is a multiple of 64K. Tiles are
// numbered with b fastest, so the blocks resident at once read neighbouring stripes of a few
// rows. Numbered with a fastest they would read every row of all four windows at once, and
// over host payload slots that loses most of the read bandwidth past about 0.4 GiB of
// windows (2^26 points) on GB10.
template<int K>
__global__ static void __launch_bounds__(256)
    pfb_unpack12_fp16_t(pfb_taps taps, int sp, size_t m1, size_t m2,
                        const __half2* __restrict__ proto, __half2* __restrict__ out) {
    constexpr int TB = 32 * K;
    __shared__ __half2 tile[2][32 * TB];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const size_t nb = m2 / TB, tiles = m1 / 32 * nb, nh = m1 * m2, half_sp = sp / 2;
    const size_t a0 = blockIdx.x / nb * 32, b0 = blockIdx.x % nb * TB;
    const size_t am = m1 - 32 - a0, bm = m2 - TB - b0; // the mirror tile
    for (int l = warp; l < 32; l += 8) {
        // The lane's K complex samples from j, and their mirrors: the K from jm, reversed.
        const size_t j = (a0 + l) * m2 + b0 + K * lane;
        const size_t jm = nh - K - j;
        const size_t pj = j / half_sp, oj = 3 * (j % half_sp);
        const size_t pm = jm / half_sp, om = 3 * (jm % half_sp);
        __half2 w0[K], w1[K], w2[K], w3[K];
        pfb_load_weights<K>(proto + j, w0);
        pfb_load_weights<K>(proto + nh + j, w1);
        pfb_load_weights<K>(proto + 2 * nh - K - j, w2);
        pfb_load_weights<K>(proto + nh - K - j, w3);
        float acc[2 * K], accm[2 * K];
#pragma unroll
        for (int i = 0; i < 2 * K; i++)
            acc[i] = accm[i] = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const uint64_t* lut = taps.lut[q];
            if (lut == nullptr)
                continue;
            int v[2 * K], vm[2 * K];
            pfb_load12<K>((const uint8_t*)(uintptr_t)lut[pj] + oj, v);
            pfb_load12<K>((const uint8_t*)(uintptr_t)lut[pm] + om, vm);
#pragma unroll
            for (int i = 0; i < K; i++) {
                const __half2 ww = q == 0   ? w0[i]
                                   : q == 1 ? w1[i]
                                   : q == 2 ? w2[K - 1 - i]
                                            : w3[K - 1 - i];
                const __half2 wm = q == 0   ? w3[i]
                                   : q == 1 ? w2[i]
                                   : q == 2 ? w1[K - 1 - i]
                                            : w0[K - 1 - i];
                acc[2 * i] += (q < 2 ? __low2float(ww) : __high2float(ww))
                              * ((v[2 * i] - 2048) * (1.0f / 2048.0f));
                acc[2 * i + 1] += (q < 2 ? __high2float(ww) : __low2float(ww))
                                  * ((v[2 * i + 1] - 2048) * (1.0f / 2048.0f));
                accm[2 * i] += (q < 2 ? __low2float(wm) : __high2float(wm))
                               * ((vm[2 * i] - 2048) * (1.0f / 2048.0f));
                accm[2 * i + 1] += (q < 2 ? __high2float(wm) : __low2float(wm))
                                   * ((vm[2 * i + 1] - 2048) * (1.0f / 2048.0f));
            }
        }
#pragma unroll
        for (int i = 0; i < K; i++) {
            tile[0][pfb_tile_index<K>(l, K * lane + i)] =
                __floats2half2_rn(acc[2 * i], acc[2 * i + 1]);
            tile[1][pfb_tile_index<K>(31 - l, K * (31 - lane) + i)] =
                __floats2half2_rn(accm[2 * i], accm[2 * i + 1]);
        }
    }
    __syncthreads();
    for (int r = warp; r < TB; r += 8) {
        out[(b0 + r) * m1 + a0 + lane] = tile[0][pfb_tile_index<K>(lane, r)];
        // A window of one tile is its own mirror.
        if (tiles > 1)
            out[(bm + r) * m1 + am + lane] = tile[1][pfb_tile_index<K>(lane, r)];
    }
}

void gpu_pfb_unpack12_fp16_t(pfb_taps taps, int samples_per_packet, size_t m1, size_t m2,
                             const __half* proto, __half2* out, cudaStream_t stream) {
    // The widest tile whose rows fit in m2 and in one packet.
    int k = 1;
    while (k < 4 && (size_t)(64 * k) <= m2 && samples_per_packet % (128 * k) == 0)
        k *= 2;
    const size_t tiles = (m1 / 32) * (m2 / (32 * k));
    const unsigned grid = tiles > 1 ? tiles / 2 : 1;
    const __half2* p = (const __half2*)proto;
    if (k == 4)
        pfb_unpack12_fp16_t<4><<<grid, 256, 0, stream>>>(taps, samples_per_packet, m1, m2, p, out);
    else if (k == 2)
        pfb_unpack12_fp16_t<2><<<grid, 256, 0, stream>>>(taps, samples_per_packet, m1, m2, p, out);
    else
        pfb_unpack12_fp16_t<1><<<grid, 256, 0, stream>>>(taps, samples_per_packet, m1, m2, p, out);
}
