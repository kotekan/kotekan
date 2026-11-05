// Timing of cudaDtvSubtract's two heavy operations per source, the lag correlations
// R_xs[j][i] = sum_t x[t][i] conj(s(t - lag_j)) and the subtraction x[t][i] -= sum_j g[j][i]
// s(t - lag_j), as the stage runs them (launch_dtv_lagcorr, launch_dtv_subtract) and as complex
// matrix products on cuBLAS: with S[t][j] = s(t - lag_j) (zero rows beyond a channel's window),
// column-major X (128 x T), G (128 x L), S (L x T):  R = X S^H  and  X -= G S, batched over the
// channels. Checks the products against the kernels, then times both at a production shape.
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
        cublasStatus_t e = (x);                                                                    \
        if (e != CUBLAS_STATUS_SUCCESS) {                                                          \
            fprintf(stderr, "cuBLAS %d at %d\n", (int)e, __LINE__);                                \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

constexpr int NI = DTV_INPUTS, HIST = DTV_HIST;

// S[c][t][j] (column-major L x T per channel: j fastest) = s[c][k][HIST + t - lag_j] for j <
// n_lag_c, 0 beyond the window and beyond the end of s (negative lags at the frame's end).
__global__ void build_s(const float2* __restrict__ s, const int* __restrict__ lags, int k,
                        int n_src, int n_times, int L, float2* __restrict__ S) {
    const int c = blockIdx.y;
    const int lag_min = lags[2 * c], nl = lags[2 * c + 1];
    const float2* sk = s + (size_t)(c * n_src + k) * (HIST + n_times);
    float2* Sc = S + (size_t)c * L * n_times;
    const size_t n = (size_t)L * n_times;
    for (size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x; q < n;
         q += (size_t)gridDim.x * blockDim.x) {
        const int j = (int)(q % L), t = (int)(q / L);
        const int idx = HIST + t - (lag_min + j);
        Sc[q] = (j < nl && idx < HIST + n_times) ? sk[idx] : make_float2(0.0f, 0.0f);
    }
}

int main(int argc, char** argv) {
    const int C = argc > 1 ? atoi(argv[1]) : 4;
    const int L = argc > 2 ? atoi(argv[2]) : 64;
    const int K = argc > 3 ? atoi(argv[3]) : 3;
    const int REPS = argc > 4 ? atoi(argv[4]) : 50;
    const int T = 8192;
    const int lag_min = L == 64 ? -16 : -4;
    std::mt19937 rng(5);
    std::normal_distribution<float> gs(0.0f, 1.0f);
    auto fill = [&](std::vector<float2>& v, float a) {
        for (auto& z : v)
            z = make_float2(a * gs(rng), a * gs(rng));
    };
    std::vector<float2> x((size_t)C * T * NI), s((size_t)C * K * (HIST + T)),
        g((size_t)C * K * L * NI);
    fill(x, 3.0f);
    fill(s, 5.0f);
    fill(g, 0.1f);
    std::vector<int> act((size_t)C * K, 2), lags;
    for (int c = 0; c < C; c++) {
        lags.push_back(lag_min);
        lags.push_back(L);
    }
    float2 *d_x, *d_x0, *d_s, *d_g, *d_rxs, *d_rss, *d_rxs2, *d_S, *d_x2;
    int *d_act, *d_lags;
    const size_t xb = x.size() * sizeof(float2), rxb = (size_t)C * K * L * NI * sizeof(float2);
    CK(cudaMalloc(&d_x, xb));
    CK(cudaMalloc(&d_x0, xb));
    CK(cudaMalloc(&d_x2, xb));
    CK(cudaMalloc(&d_s, s.size() * sizeof(float2)));
    CK(cudaMalloc(&d_g, g.size() * sizeof(float2)));
    CK(cudaMalloc(&d_rxs, rxb));
    CK(cudaMalloc(&d_rxs2, rxb));
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
    const cuComplex one = {1.0f, 0.0f}, zero = {0.0f, 0.0f}, mone = {-1.0f, 0.0f};

    // one frame's K sources, the stage's way
    auto run_kernels = [&]() {
        for (int k = 0; k < K; k++) {
            launch_dtv_lagcorr(d_x, d_s, d_act, d_lags, k, C, K, T, L, d_rxs, d_rss, st);
            launch_dtv_subtract(d_x, d_s, d_g, d_act, d_lags, k, C, K, T, L, st);
        }
    };
    // and as products: S for source k, R = X S^H into its block, X -= G S
    auto run_gemm = [&]() {
        for (int k = 0; k < K; k++) {
            build_s<<<dim3(256, C), 256, 0, st>>>(d_s, d_lags, k, K, T, L, d_S);
            CB(cublasCgemmStridedBatched(
                h, CUBLAS_OP_N, CUBLAS_OP_C, NI, L, T, &one, (const cuComplex*)d_x2, NI,
                (long long)T * NI, (const cuComplex*)d_S, L, (long long)L * T, &zero,
                (cuComplex*)(d_rxs2 + (size_t)k * L * NI), NI, (long long)K * L * NI, C));
            CB(cublasCgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, NI, T, L, &mone,
                                         (const cuComplex*)(d_g + (size_t)k * L * NI), NI,
                                         (long long)K * L * NI, (const cuComplex*)d_S, L,
                                         (long long)L * T, &one, (cuComplex*)d_x2, NI,
                                         (long long)T * NI, C));
        }
    };

    // correctness: one frame each from the same x
    CK(cudaMemcpy(d_x, d_x0, xb, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(d_x2, d_x0, xb, cudaMemcpyDeviceToDevice));
    CK(cudaMemset(d_rxs, 0, rxb));
    CK(cudaMemset(d_rss, 0, (size_t)C * K * L * sizeof(float2)));
    run_kernels();
    run_gemm();
    CK(cudaStreamSynchronize(st));
    std::vector<float2> r1((size_t)C * K * L * NI), r2(r1.size()), x1(x.size()), x2(x.size());
    CK(cudaMemcpy(r1.data(), d_rxs, rxb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r2.data(), d_rxs2, rxb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(x1.data(), d_x, xb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(x2.data(), d_x2, xb, cudaMemcpyDeviceToHost));
    double er = 0, rr = 0, ex = 0, xx = 0;
    for (size_t q = 0; q < r1.size(); q++) {
        er = std::max(er, (double)std::hypot(r1[q].x - r2[q].x, r1[q].y - r2[q].y));
        rr = std::max(rr, (double)std::hypot(r1[q].x, r1[q].y));
    }
    for (size_t q = 0; q < x1.size(); q++) {
        ex = std::max(ex, (double)std::hypot(x1[q].x - x2[q].x, x1[q].y - x2[q].y));
        xx = std::max(xx, (double)std::hypot(x1[q].x, x1[q].y));
    }
    printf("C %d channels, %d lags, %d sources, %d samples: products vs kernels: R_xs max |d| / "
           "max |R| %.2e, residual max |d| / max |x| %.2e\n",
           C, L, K, T, er / rr, ex / xx);

    // timing, one frame = K sources
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
    const float tk = timeit(run_kernels);
    const float tg = timeit(run_gemm);
    CB(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    const float tt = timeit(run_gemm);
    // TF32 accuracy against the FP32 products
    CK(cudaMemcpy(d_x2, d_x0, xb, cudaMemcpyDeviceToDevice));
    run_gemm();
    CK(cudaStreamSynchronize(st));
    std::vector<float2> r3(r1.size()), x3(x.size());
    CK(cudaMemcpy(r3.data(), d_rxs2, rxb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(x3.data(), d_x2, xb, cudaMemcpyDeviceToHost));
    double e3 = 0, ex3 = 0;
    for (size_t q = 0; q < r1.size(); q++)
        e3 = std::max(e3, (double)std::hypot(r1[q].x - r3[q].x, r1[q].y - r3[q].y));
    for (size_t q = 0; q < x1.size(); q++)
        ex3 = std::max(ex3, (double)std::hypot(x1[q].x - x3[q].x, x1[q].y - x3[q].y));
    // where the TF32 time goes: S, the correlation product, the subtraction product (one source)
    auto t_build =
        timeit([&]() { build_s<<<dim3(256, C), 256, 0, st>>>(d_s, d_lags, 0, K, T, L, d_S); });
    auto t_corr = timeit([&]() {
        CB(cublasCgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_C, NI, L, T, &one,
                                     (const cuComplex*)d_x2, NI, (long long)T * NI,
                                     (const cuComplex*)d_S, L, (long long)L * T, &zero,
                                     (cuComplex*)d_rxs2, NI, (long long)K * L * NI, C));
    });
    auto t_sub = timeit([&]() {
        CB(cublasCgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, NI, T, L, &mone,
                                     (const cuComplex*)d_g, NI, (long long)K * L * NI,
                                     (const cuComplex*)d_S, L, (long long)L * T, &one,
                                     (cuComplex*)d_x2, NI, (long long)T * NI, C));
    });
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const double mb_x = (double)C * T * NI * 8 / 1e6, mb_s = (double)C * L * T * 8 / 1e6;
    printf("%s, %d SMs: per source TF32 build S %.3f ms (writes %.0f MB: %.0f GB/s), R = X S^H "
           "%.3f ms (reads %.0f MB), X -= G S %.3f ms (reads/writes %.0f MB: %.0f GB/s)\n",
           prop.name, prop.multiProcessorCount, t_build, mb_s, mb_s / t_build, t_corr, mb_x + mb_s,
           t_sub, 2 * mb_x + mb_s, (2 * mb_x + mb_s) / t_sub);
    const double gflop = 2.0 * 8.0 * NI * (double)T * L * K * C / 1e9;
    printf("per frame (%d sources x %d channels, %.2f GFLOP): kernels %.3f ms (%.1f TFLOP/s), "
           "products FP32 %.3f ms (%.1f), TF32 %.3f ms (%.1f; R_xs %.1e, residual %.1e rel)\n",
           K, C, gflop, tk, gflop / tk, tg, gflop / tg, tt, gflop / tt, e3 / rr, ex3 / xx);
    return 0;
}
