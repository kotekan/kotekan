// Self-checking test for gpu_tensor_fft: seeded noise + a strong tone through the
// tensor-core FFT, checked against a direct double-precision DFT on a few transforms per
// size and against batched cuFFT fp16 over the whole batch.

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "cudaUtils.hpp"       // for CHECK_CUDA_ERROR
#include "errors.h"            // for TEST_PASSED, TEST_FAILED
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "tensorFFT.hpp"       // for gpu_tensor_fft

#include <cmath>       // for cos, sin, sqrt
#include <complex>     // for complex, norm, polar
#include <cuda_fp16.h> // for __half2
#include <cufft.h>     // for cufftCreate, cufftDestroy
#include <cufftXt.h>   // for cufftXtMakePlanMany, cufftXtExec
#include <random>      // for mt19937, normal_distribution
#include <string>      // for string
#include <vector>      // for vector

class testTensorFFT : public kotekan::Stage {
public:
    testTensorFFT(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    ~testTensorFFT() override = default;

    // Direct DFT of one transform read back from fp16, in double.
    static double rel_l2_vs_dft(const std::vector<__half2>& in, const std::vector<__half2>& out,
                                size_t base, size_t n) {
        double num = 0, den = 0;
        for (size_t k = 0; k < n; k++) {
            std::complex<double> ref = 0;
            for (size_t j = 0; j < n; j++) {
                const std::complex<double> x(__low2float(in[base + j]), __high2float(in[base + j]));
                ref += x * std::polar(1.0, -2.0 * M_PI * (double)(j * k % n) / (double)n);
            }
            const std::complex<double> y(__low2float(out[base + k]), __high2float(out[base + k]));
            num += std::norm(y - ref);
            den += std::norm(ref);
        }
        return std::sqrt(num / den);
    }

    bool run_case(size_t n, size_t batch) {
        std::mt19937 rng(12345 + n);
        std::normal_distribution<float> noise(0.0f, 0.25f);
        const size_t total = n * batch;
        const size_t tone_bin = n / 5 + 1;
        std::vector<__half2> h_in(total);
        for (size_t b = 0; b < batch; b++) {
            const double phase = 0.37 * (double)b;
            for (size_t j = 0; j < n; j++) {
                const double ang = 2.0 * M_PI * (double)(j * tone_bin % n) / (double)n + phase;
                h_in[b * n + j] = __floats2half2_rn(noise(rng) + 4.0f * (float)cos(ang),
                                                    noise(rng) + 4.0f * (float)sin(ang));
            }
        }

        __half2 *d_in, *d_out, *d_cu;
        CHECK_CUDA_ERROR(cudaMalloc(&d_in, total * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_out, total * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_cu, total * sizeof(__half2)));
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_in, h_in.data(), total * sizeof(__half2), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_out, 0xee, total * sizeof(__half2)));

        gpu_tensor_fft(d_in, d_out, n, batch, nullptr);
        CHECK_CUDA_ERROR(cudaGetLastError());

        cufftHandle plan;
        long long nn = n;
        size_t ws;
        if (cufftCreate(&plan) != CUFFT_SUCCESS
            || cufftXtMakePlanMany(plan, 1, &nn, nullptr, 1, nn, CUDA_C_16F, nullptr, 1, nn,
                                   CUDA_C_16F, batch, &ws, CUDA_C_16F)
                   != CUFFT_SUCCESS
            || cufftXtExec(plan, d_in, d_cu, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            ERROR("cuFFT reference plan failed at n={:d}", n);
            return false;
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cufftDestroy(plan);

        std::vector<__half2> h_out(total), h_cu(total);
        CHECK_CUDA_ERROR(
            cudaMemcpy(h_out.data(), d_out, total * sizeof(__half2), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(
            cudaMemcpy(h_cu.data(), d_cu, total * sizeof(__half2), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaFree(d_in));
        CHECK_CUDA_ERROR(cudaFree(d_out));
        CHECK_CUDA_ERROR(cudaFree(d_cu));

        // Direct DFT on a few transforms spread over the batch, cuFFT on all of them.
        const int ncheck = n >= 4096 ? 4 : 8;
        double worst = 0;
        for (int c = 0; c < ncheck; c++)
            worst = std::max(
                worst, rel_l2_vs_dft(h_in, h_out, (size_t)c * (batch - 1) / (ncheck - 1) * n, n));
        double worst_x = 0;
        for (size_t b = 0; b < batch; b++) {
            double num = 0, den = 0;
            for (size_t j = 0; j < n; j++) {
                const std::complex<double> y(__low2float(h_out[b * n + j]),
                                             __high2float(h_out[b * n + j]));
                const std::complex<double> r(__low2float(h_cu[b * n + j]),
                                             __high2float(h_cu[b * n + j]));
                num += std::norm(y - r);
                den += std::norm(r);
            }
            worst_x = std::max(worst_x, std::sqrt(num / den));
        }

        const double limit = n <= 1024 ? 0.003 : 0.005;
        INFO("n={:d} batch={:d}: rel L2 vs DFT {:.4f}% (limit {:.1f}%), vs cuFFT fp16 {:.4f}%", n,
             batch, 100 * worst, 100 * limit, 100 * worst_x);
        if (worst > limit || worst_x > 0.01) {
            ERROR("n={:d}: accuracy outside limits", n);
            return false;
        }
        return true;
    }

    void main_thread() override {
        bool ok = run_case(32, 65536);
        ok = run_case(64, 4096) && ok;
        ok = run_case(256, 8192) && ok;
        ok = run_case(1024, 2048) && ok;
        ok = run_case(4096, 512) && ok;
        ok = run_case(8192, 257) && ok;
        if (ok)
            TEST_PASSED();
        else
            TEST_FAILED();
    }
};

REGISTER_KOTEKAN_STAGE(testTensorFFT);
