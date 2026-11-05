// Timing of cudaDtvSubtract's per-frame sequence (shift, decode with source 0's timestream, per
// source R_ss and the fused pass, re-quantisation) on random 4+4b voltages at a production shape
// (8192 samples, 128 inputs), with every source subtracting. The last n_small channels have a
// 16-lag window (-4..+11) and one source, as MUOS's. Channel 0 and the last channel are checked
// against a CPU sum in double: each timestream, R_xs and the residual before re-quantising.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 -o dtv_gemm_bench dtv_gemm_bench.cu \
//        <repo>/lib/cuda/cudaDtvSubtractKernel.cu -I<repo>/lib/cuda
//   dtv_gemm_bench [n_chan 4] [n_lag 64] [n_src 3] [reps 50] [n_small 0]

#include "cudaDtvSubtractKernel.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace gnss_cuda;
using cd = std::complex<double>;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,           \
                    __LINE__);                                                                     \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

constexpr int NI = DTV_INPUTS, HIST = DTV_HIST, T = 8192;

static cd dec44(unsigned char b) {
    return cd((double)((b >> 4) & 0x0f) - 8.0, (double)(b & 0x0f) - 8.0);
}

int main(int argc, char** argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 4;
    const int L = argc > 2 ? atoi(argv[2]) : 64;
    const int K = argc > 3 ? atoi(argv[3]) : 3;
    const int REPS = argc > 4 ? atoi(argv[4]) : 50;
    const int SMALL = argc > 5 ? atoi(argv[5]) : 0;
    if (C < 1 || L < 16 || L > DTV_MAX_LAGS || K < 1 || K > DTV_MAX_SRC || SMALL > C) {
        fprintf(stderr, "bad shape\n");
        return 2;
    }
    const int L16 = dtv_l16(L);
    std::vector<int> lags, src(C), act((size_t)C * K, 0), sel(C);
    for (int c = 0; c < C; c++) {
        const bool small = c >= C - SMALL;
        lags.push_back(small ? -4 : (L == 64 ? -16 : -8));
        lags.push_back(small ? 16 : L);
        src[c] = small ? 1 : K;
        for (int k = 0; k < src[c]; k++)
            act[(size_t)c * K + k] = 2;
        sel[c] = c;
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<unsigned char> frame((size_t)T * C * NI);
    for (auto& b : frame) {
        const int re = std::clamp((int)std::lround(3.0f * gauss(rng)), -7, 7);
        const int im = std::clamp((int)std::lround(3.0f * gauss(rng)), -7, 7);
        b = (unsigned char)(((re + 8) << 4) | (im + 8));
    }
    std::vector<float2> w((size_t)C * K * NI), g((size_t)C * K * L * NI, make_float2(0, 0));
    std::vector<float> gs((size_t)C * K * 2 * L16 * NI, 0.0f);
    for (auto& z : w)
        z = make_float2(gauss(rng) / 16.0f, gauss(rng) / 16.0f);
    for (int c = 0; c < C; c++)
        for (int k = 0; k < src[c]; k++)
            for (int j = 0; j < lags[2 * c + 1]; j++)
                for (int i = 0; i < NI; i++) {
                    const float2 v = make_float2(0.05f * gauss(rng), 0.05f * gauss(rng));
                    const size_t ck = (size_t)c * K + k;
                    g[(ck * L + j) * NI + i] = v;
                    gs[(ck * 2 * L16 + j) * NI + i] = v.x;
                    gs[((ck * 2 + 1) * L16 + j) * NI + i] = v.y;
                }

    const DtvOutLayout lay = dtv_out_layout(C, K, L, true);
    unsigned char *d_frame0, *d_frame, *d_out;
    int *d_sel, *d_act, *d_lags;
    float2 *d_s, *d_w;
    float *d_x, *d_gs;
    const size_t sb = (size_t)C * K * (HIST + T) * sizeof(float2);
    CK(cudaMalloc(&d_frame0, frame.size()));
    CK(cudaMalloc(&d_frame, frame.size()));
    CK(cudaMalloc(&d_out, lay.bytes));
    CK(cudaMalloc(&d_sel, C * sizeof(int)));
    CK(cudaMalloc(&d_act, act.size() * sizeof(int)));
    CK(cudaMalloc(&d_lags, lags.size() * sizeof(int)));
    CK(cudaMalloc(&d_s, sb));
    CK(cudaMalloc(&d_w, w.size() * sizeof(float2)));
    CK(cudaMalloc(&d_x, (size_t)C * 2 * T * NI * sizeof(float)));
    CK(cudaMalloc(&d_gs, gs.size() * sizeof(float)));
    CK(cudaMemcpy(d_frame0, frame.data(), frame.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sel, sel.data(), C * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_act, act.data(), act.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gs, gs.data(), gs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_s, 0, sb));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    float2* rxs = (float2*)(d_out + lay.off_rxs);
    float* ps = (float*)(d_out + lay.off_ps);

    // the stage's per-frame sequence, without the header and the rolling sums
    auto frame_pass = [&]() {
        CK(cudaMemsetAsync(d_out + lay.off_ps, 0, lay.bytes - lay.off_ps, st));
        launch_dtv_shift(d_s, C, K, T, st);
        launch_dtv_decode(d_frame, d_sel, d_w, d_act, C, K, C, T, d_x,
                          (float*)(d_out + lay.off_pw_raw), d_s, ps, st);
        for (int k = 0; k < K; k++) {
            launch_dtv_rss(d_s, d_act, d_lags, k, C, K, T, L, (float2*)(d_out + lay.off_rss), st);
            launch_dtv_fused(d_x, d_s, d_gs, d_w, d_act, d_lags, k, C, K, T, L, rxs, ps, st);
        }
        launch_dtv_encode(d_x, d_sel, d_act, C, K, C, T, d_frame, (float*)(d_out + lay.off_pw_res),
                          st);
    };

    // one frame from a zero history, then the CPU check
    CK(cudaMemcpy(d_frame, d_frame0, frame.size(), cudaMemcpyDeviceToDevice));
    frame_pass();
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    std::vector<float> xg((size_t)C * 2 * T * NI);
    std::vector<float2> sg((size_t)C * K * (HIST + T)), rg((size_t)C * K * L * NI);
    CK(cudaMemcpy(xg.data(), d_x, xg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sg.data(), d_s, sb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rg.data(), rxs, rg.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    for (int c : {0, C - 1}) {
        const int lag_min = lags[2 * c], nl = lags[2 * c + 1];
        std::vector<cd> x((size_t)T * NI), s(HIST + T);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < NI; i++)
                x[(size_t)t * NI + i] = dec44(frame[((size_t)t * C + c) * NI + i]);
        double es = 0, ms = 0, er = 0, mr = 0;
        for (int k = 0; k < src[c]; k++) {
            const size_t ck = (size_t)c * K + k;
            std::fill(s.begin(), s.end(), cd(0, 0));
            for (int t = 0; t < T; t++) {
                cd b(0, 0);
                for (int i = 0; i < NI; i++)
                    b += cd(w[ck * NI + i].x, w[ck * NI + i].y) * x[(size_t)t * NI + i];
                s[HIST + t] = b;
                const float2 d = sg[ck * (HIST + T) + HIST + t];
                es = std::max(es, std::abs(b - cd(d.x, d.y)));
                ms = std::max(ms, std::abs(b));
            }
            for (int j = 0; j < nl; j += 7)
                for (int i = 0; i < NI; i += 5) {
                    cd a(0, 0);
                    for (int t = 0; t < T; t++) {
                        const int idx = HIST + t - (lag_min + j);
                        if (idx < HIST + T)
                            a += x[(size_t)t * NI + i] * std::conj(s[idx]);
                    }
                    const float2 d = rg[(ck * L + j) * NI + i];
                    er = std::max(er, std::abs(a - cd(d.x, d.y)));
                    mr = std::max(mr, std::abs(a));
                }
            std::vector<cd> r(x);
            for (int t = 0; t < T; t++)
                for (int j = 0; j < nl; j++) {
                    const int idx = HIST + t - (lag_min + j);
                    if (idx >= HIST + T)
                        continue;
                    for (int i = 0; i < NI; i++)
                        r[(size_t)t * NI + i] -=
                            cd(g[(ck * L + j) * NI + i].x, g[(ck * L + j) * NI + i].y) * s[idx];
                }
            x.swap(r);
        }
        double ex = 0, mx = 0;
        for (int t = 0; t < T; t++)
            for (int i = 0; i < NI; i++) {
                const cd v(xg[((size_t)c * 2 * T + t) * NI + i],
                           xg[((size_t)(c * 2 + 1) * T + t) * NI + i]);
                ex = std::max(ex, std::abs(v - x[(size_t)t * NI + i]));
                mx = std::max(mx, std::abs(x[(size_t)t * NI + i]));
            }
        printf("channel %d (%d lags, %d sources) vs CPU: timestreams %.1e, R_xs %.1e (of the max), "
               "residual max |error| %.1e (max |r| %.1f)\n",
               c, nl, src[c], es / ms, er / mr, ex, mx);
    }

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto timeit = [&](auto&& f) {
        f();
        CK(cudaEventRecord(e0, st));
        for (int r = 0; r < REPS; r++)
            f();
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return ms / REPS;
    };
    const float t_frame = timeit(frame_pass);
    const float t_pass = timeit([&]() {
        launch_dtv_fused(d_x, d_s, d_gs, d_w, d_act, d_lags, 0, C, K, T, L, rxs, ps, st);
    });
    const float t_edge = timeit([&]() {
        launch_dtv_decode(d_frame, d_sel, d_w, d_act, C, K, C, T, d_x,
                          (float*)(d_out + lay.off_pw_raw), d_s, ps, st);
        launch_dtv_encode(d_x, d_sel, d_act, C, K, C, T, d_frame, (float*)(d_out + lay.off_pw_res),
                          st);
    });
    int n_sc = 0;
    for (int c = 0; c < C; c++)
        n_sc += src[c];
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const double mb = (double)C * T * NI * 8 / 1e6; // x, MB
    printf("%s: %d channels (%d at 16 lags x 1 source), %d x %d lags, %d source-channels: %.3f ms "
           "per frame (decode + encode %.3f ms); one fused pass %.3f ms = %.0f GB/s over x read + "
           "written\n",
           prop.name, C, SMALL, K, L, n_sc, t_frame, t_edge, t_pass, 2 * mb / t_pass);
    return 0;
}
