// Timing of cudaDtvSubtract's per-source work: the lagged timestream with R_ss
// (launch_dtv_lagged), the lag correlations (dtv_lagcorr) and the subtraction (dtv_subtract), as
// complex matrix products on cuBLAS in FP32 and TF32, with the TF32 error against FP32, at a
// production shape (8192 samples, 128 inputs). The kernels these products replaced took, on
// cf05's L40S per frame of 3 sources x 64 lags, 2.0 ms for 4 channels and 5.0 ms for 14.
//
//   nvcc -O3 -std=c++17 -arch=native -o dtv_gemm_bench dtv_gemm_bench.cu \
//        <repo>/lib/cuda/cudaDtvSubtractKernel.cu -I<repo>/lib/cuda -lcublas
//   dtv_gemm_bench [n_chan 4] [n_lag 64] [n_src 3] [reps 50]

#include "cudaDtvSubtractKernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace gnss_cuda;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__);                   \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)
#define CB(x)                                                                                      \
    do {                                                                                           \
        if ((x) != CUBLAS_STATUS_SUCCESS) {                                                        \
            fprintf(stderr, "cuBLAS failed at %d\n", __LINE__);                                    \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

constexpr int NI = DTV_INPUTS, HIST = DTV_HIST;

int main(int argc, char** argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 4;
    const int L = argc > 2 ? atoi(argv[2]) : 64;
    const int K = argc > 3 ? atoi(argv[3]) : 3;
    const int REPS = argc > 4 ? atoi(argv[4]) : 50;
    const int T = 8192;
    const int lag_min = L == 64 ? -16 : -4;
    std::mt19937 rng(5);
    std::normal_distribution<float> gs(0.0f, 1.0f);
    std::vector<float2> x((size_t)C * T * NI), s((size_t)C * K * (HIST + T)),
        g((size_t)C * K * L * NI);
    for (auto* v : {&x, &s, &g})
        for (auto& z : *v)
            z = make_float2(gs(rng), gs(rng));
    std::vector<int> act((size_t)C * K, 2), lags;
    for (int c = 0; c < C; c++) {
        lags.push_back(lag_min);
        lags.push_back(L);
    }
    float2 *d_x0, *d_x, *d_s, *d_g, *d_rxs, *d_rss, *d_S;
    int *d_act, *d_lags;
    const size_t xb = x.size() * sizeof(float2), rxb = (size_t)C * K * L * NI * sizeof(float2);
    CK(cudaMalloc(&d_x0, xb));
    CK(cudaMalloc(&d_x, xb));
    CK(cudaMalloc(&d_s, s.size() * sizeof(float2)));
    CK(cudaMalloc(&d_g, g.size() * sizeof(float2)));
    CK(cudaMalloc(&d_rxs, rxb));
    CK(cudaMalloc(&d_rss, (size_t)C * K * L * sizeof(float2)));
    CK(cudaMalloc(&d_S, (size_t)C * L * T * sizeof(float2)));
    CK(cudaMalloc(&d_act, act.size() * sizeof(int)));
    CK(cudaMalloc(&d_lags, lags.size() * sizeof(int)));
    CK(cudaMemcpy(d_x0, x.data(), xb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s, s.data(), s.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_g, g.data(), g.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_act, act.data(), act.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(int), cudaMemcpyHostToDevice));
    cublasHandle_t h;
    CB(cublasCreate(&h));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    CB(cublasSetStream(h, st));

    auto lagged = [&](int k) {
        launch_dtv_lagged(d_s, d_act, d_lags, k, C, K, T, L, d_S, d_rss, st);
    };
    auto corr = [&](int k) { CB(dtv_lagcorr(h, d_x, d_S, k, C, K, T, L, d_rxs)); };
    auto sub = [&](int k) { CB(dtv_subtract(h, d_x, d_S, d_g, k, C, K, T, L)); };
    auto frame = [&]() {
        for (int k = 0; k < K; k++) {
            lagged(k);
            corr(k);
            sub(k);
        }
    };
    auto result = [&](std::vector<float2>& r, std::vector<float2>& xr) {
        CK(cudaMemcpy(d_x, d_x0, xb, cudaMemcpyDeviceToDevice));
        frame();
        CK(cudaStreamSynchronize(st));
        r.resize((size_t)C * K * L * NI);
        xr.resize(x.size());
        CK(cudaMemcpy(r.data(), d_rxs, rxb, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(xr.data(), d_x, xb, cudaMemcpyDeviceToHost));
    };
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

    std::vector<float2> r32, x32, rtf, xtf;
    result(r32, x32);
    const float t32 = timeit(frame);
    CB(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    result(rtf, xtf);
    const float ttf = timeit(frame);
    const float t_lag = timeit([&]() { lagged(0); }), t_corr = timeit([&]() { corr(0); }),
                t_sub = timeit([&]() { sub(0); });
    double er = 0, rr = 0, ex = 0, xx = 0;
    for (size_t q = 0; q < r32.size(); q++) {
        er = std::max(er, (double)std::hypot(r32[q].x - rtf[q].x, r32[q].y - rtf[q].y));
        rr = std::max(rr, (double)std::hypot(r32[q].x, r32[q].y));
    }
    for (size_t q = 0; q < x32.size(); q++) {
        ex = std::max(ex, (double)std::hypot(x32[q].x - xtf[q].x, x32[q].y - xtf[q].y));
        xx = std::max(xx, (double)std::hypot(x32[q].x, x32[q].y));
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const double gflop = 2.0 * 8.0 * NI * (double)T * L * K * C / 1e9;
    printf("%s: %d channels x %d sources x %d lags (%.1f GFLOP per frame): FP32 %.3f ms, TF32 %.3f "
           "ms (%.1f TFLOP/s; per source: lagged + R_ss %.3f, R = X S^H %.3f, X -= G S %.3f ms); "
           "TF32 vs FP32: R_xs %.1e, residual %.1e of the largest\n",
           prop.name, C, K, L, gflop, t32, ttf, gflop / ttf, t_lag, t_corr, t_sub, er / rr,
           ex / xx);
    return 0;
}
