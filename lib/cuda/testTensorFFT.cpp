// Self-checking accuracy test for the batched F-engine FFTs: seeded noise + a strong tone
// through the fp16 and TF32 tensor-core FFTs and the fp16 and fp32 cuFFT plans, each
// checked against a double-precision FFT on a few transforms per size.
// With perf: true it also times each implementation at batches that leave L2.

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "cudaUtils.hpp"       // for CHECK_CUDA_ERROR
#include "errors.h"            // for TEST_PASSED, TEST_FAILED
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "tensorFFT.hpp"       // for gpu_tensor_fft, gpu_tensor_fft_f32

#include <cmath>       // for cos, sin, sqrt
#include <complex>     // for complex, norm, polar
#include <cuda_fp16.h> // for __half2
#include <cufft.h>     // for cufftCreate, cufftDestroy, cufftExecC2C
#include <cufftXt.h>   // for cufftXtMakePlanMany, cufftXtExec
#include <random>      // for mt19937, normal_distribution
#include <string>      // for string
#include <vector>      // for vector

class testTensorFFT : public kotekan::Stage {
public:
    testTensorFFT(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        perf(config.get_default<bool>(unique_name, "perf", false)) {}

    ~testTensorFFT() override = default;

    // Radix-2 FFT of the transform at base, in double, from the exact fp16 input values.
    static std::vector<std::complex<double>> fft_ref(const std::vector<__half2>& in, size_t base,
                                                     size_t n) {
        std::vector<std::complex<double>> x(n);
        for (size_t j = 0, r = 0; j < n; j++) {
            x[r] = std::complex<double>(__low2float(in[base + j]), __high2float(in[base + j]));
            size_t bit = n >> 1; // r = bit-reversed j: increment from the top bit down
            for (; r & bit; bit >>= 1)
                r ^= bit;
            r |= bit;
        }
        for (size_t len = 2; len <= n; len <<= 1)
            for (size_t k = 0; k < len / 2; k++) {
                const std::complex<double> w =
                    std::polar(1.0, -2.0 * M_PI * (double)k / (double)len);
                for (size_t i = k; i < n; i += len) {
                    const std::complex<double> u = x[i], v = x[i + len / 2] * w;
                    x[i] = u + v;
                    x[i + len / 2] = u - v;
                }
            }
        return x;
    }

    template<typename T2>
    static double rel_l2(const std::vector<T2>& out, size_t base,
                         const std::vector<std::complex<double>>& ref) {
        double num = 0, den = 0;
        for (size_t k = 0; k < ref.size(); k++) {
            const std::complex<double> y(re(out[base + k]), im(out[base + k]));
            num += std::norm(y - ref[k]);
            den += std::norm(ref[k]);
        }
        return std::sqrt(num / den);
    }

    static double re(const __half2 v) {
        return __low2float(v);
    }
    static double im(const __half2 v) {
        return __high2float(v);
    }
    static double re(const float2 v) {
        return v.x;
    }
    static double im(const float2 v) {
        return v.y;
    }

    bool run_case(size_t n, size_t batch) {
        std::mt19937 rng(12345 + n);
        std::normal_distribution<float> noise(0.0f, 0.25f);
        const size_t total = n * batch;
        const size_t tone_bin = n / 5 + 1;
        std::vector<__half2> h_in(total);
        std::vector<float2> h_in32(total);
        for (size_t b = 0; b < batch; b++) {
            const double phase = 0.37 * (double)b;
            for (size_t j = 0; j < n; j++) {
                const double ang = 2.0 * M_PI * (double)(j * tone_bin % n) / (double)n + phase;
                const __half2 v = __floats2half2_rn(noise(rng) + 4.0f * (float)cos(ang),
                                                    noise(rng) + 4.0f * (float)sin(ang));
                h_in[b * n + j] = v;
                // The float input holds the exact fp16 values, so every path sees the same
                // signal and the double reference is common.
                h_in32[b * n + j] = make_float2(__low2float(v), __high2float(v));
            }
        }

        __half2 *d_in, *d_t16, *d_cu16;
        float2 *d_in32, *d_t32, *d_cu32;
        CHECK_CUDA_ERROR(cudaMalloc(&d_in, total * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_t16, total * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_cu16, total * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_in32, total * sizeof(float2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_t32, total * sizeof(float2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_cu32, total * sizeof(float2)));
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_in, h_in.data(), total * sizeof(__half2), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_in32, h_in32.data(), total * sizeof(float2), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_t16, 0xee, total * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMemset(d_t32, 0xee, total * sizeof(float2)));

        gpu_tensor_fft(d_in, d_t16, n, batch, nullptr);
        gpu_tensor_fft_f32(d_in32, d_t32, n, batch, nullptr);
        CHECK_CUDA_ERROR(cudaGetLastError());

        cufftHandle plan16, plan32;
        long long nn = n;
        size_t ws;
        if (cufftCreate(&plan16) != CUFFT_SUCCESS
            || cufftXtMakePlanMany(plan16, 1, &nn, nullptr, 1, nn, CUDA_C_16F, nullptr, 1, nn,
                                   CUDA_C_16F, batch, &ws, CUDA_C_16F)
                   != CUFFT_SUCCESS
            || cufftXtExec(plan16, d_in, d_cu16, CUFFT_FORWARD) != CUFFT_SUCCESS
            || cufftPlan1d(&plan32, (int)n, CUFFT_C2C, (int)batch) != CUFFT_SUCCESS
            || cufftExecC2C(plan32, (cufftComplex*)d_in32, (cufftComplex*)d_cu32, CUFFT_FORWARD)
                   != CUFFT_SUCCESS) {
            ERROR("cuFFT reference plans failed at n={:d}", n);
            return false;
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Event-timed reps of each implementation, for the record alongside the accuracy.
        if (perf) {
            cudaEvent_t t0, t1;
            CHECK_CUDA_ERROR(cudaEventCreate(&t0));
            CHECK_CUDA_ERROR(cudaEventCreate(&t1));
            const int reps = 10;
            // read + write bytes, over ms: bytes * 1e-6 / ms = GB/s.
            const double mb16 = 2.0 * total * sizeof(__half2) / 1e6;
            const double mb32 = 2.0 * total * sizeof(float2) / 1e6;
            float ms[4];
            for (int impl = 0; impl < 4; impl++) {
                CHECK_CUDA_ERROR(cudaEventRecord(t0));
                for (int r = 0; r < reps; r++) {
                    if (impl == 0)
                        gpu_tensor_fft(d_in, d_t16, n, batch, nullptr);
                    else if (impl == 1)
                        cufftXtExec(plan16, d_in, d_cu16, CUFFT_FORWARD);
                    else if (impl == 2)
                        gpu_tensor_fft_f32(d_in32, d_t32, n, batch, nullptr);
                    else
                        cufftExecC2C(plan32, (cufftComplex*)d_in32, (cufftComplex*)d_cu32,
                                     CUFFT_FORWARD);
                }
                CHECK_CUDA_ERROR(cudaEventRecord(t1));
                CHECK_CUDA_ERROR(cudaEventSynchronize(t1));
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms[impl], t0, t1));
                ms[impl] /= reps;
            }
            CHECK_CUDA_ERROR(cudaEventDestroy(t0));
            CHECK_CUDA_ERROR(cudaEventDestroy(t1));
            INFO("n={:d} batch={:d}: ms/exec, tensor fp16 {:.3f} ({:.0f} GB/s), cuFFT fp16 {:.3f} "
                 "({:.0f} GB/s), tensor tf32 {:.3f} ({:.0f} GB/s), cuFFT fp32 {:.3f} ({:.0f} GB/s)",
                 n, batch, ms[0], mb16 / ms[0], ms[1], mb16 / ms[1], ms[2], mb32 / ms[2], ms[3],
                 mb32 / ms[3]);
        }
        cufftDestroy(plan16);
        cufftDestroy(plan32);

        std::vector<__half2> h_t16(total), h_cu16(total);
        std::vector<float2> h_t32(total), h_cu32(total);
        CHECK_CUDA_ERROR(
            cudaMemcpy(h_t16.data(), d_t16, total * sizeof(__half2), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(
            cudaMemcpy(h_cu16.data(), d_cu16, total * sizeof(__half2), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(
            cudaMemcpy(h_t32.data(), d_t32, total * sizeof(float2), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(
            cudaMemcpy(h_cu32.data(), d_cu32, total * sizeof(float2), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaFree(d_in));
        CHECK_CUDA_ERROR(cudaFree(d_t16));
        CHECK_CUDA_ERROR(cudaFree(d_cu16));
        CHECK_CUDA_ERROR(cudaFree(d_in32));
        CHECK_CUDA_ERROR(cudaFree(d_t32));
        CHECK_CUDA_ERROR(cudaFree(d_cu32));

        // Check a few transforms spread over the batch.
        const int ncheck = n >= 4096 ? 4 : 8;
        double w_t16 = 0, w_cu16 = 0, w_t32 = 0, w_cu32 = 0;
        for (int c = 0; c < ncheck; c++) {
            const size_t base = (size_t)c * (batch - 1) / (ncheck - 1) * n;
            const auto ref = fft_ref(h_in, base, n);
            w_t16 = std::max(w_t16, rel_l2(h_t16, base, ref));
            w_cu16 = std::max(w_cu16, rel_l2(h_cu16, base, ref));
            w_t32 = std::max(w_t32, rel_l2(h_t32, base, ref));
            w_cu32 = std::max(w_cu32, rel_l2(h_cu32, base, ref));
        }

        const double limit16 = n <= 1024 ? 0.003 : 0.005;
        const double limit32 = 0.0005; // TF32 keeps fp32 storage: well under the fp16 paths
        INFO("n={:d} batch={:d}: rel L2 vs DFT, tensor fp16 {:.4f}% (limit {:.1f}%), cuFFT fp16 "
             "{:.4f}%, tensor tf32 {:.5f}% (limit {:.2f}%), cuFFT fp32 {:.5f}%",
             n, batch, 100 * w_t16, 100 * limit16, 100 * w_cu16, 100 * w_t32, 100 * limit32,
             100 * w_cu32);
        if (w_t16 > limit16 || w_t32 > limit32) {
            ERROR("n={:d}: accuracy outside limits", n);
            return false;
        }
        return true;
    }

    // gpu_reorder_f32 round trip: bin order -> transform-local -> bin order must be the
    // identity, and the tone spot-check pins the mapping direction.
    bool check_reorder(size_t m1, size_t m2) {
        const size_t m = m1 * m2;
        std::vector<float> h(m);
        for (size_t i = 0; i < m; i++)
            h[i] = (float)(i % 1000003);
        float *d_a, *d_b;
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, m * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, m * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_a, h.data(), m * sizeof(float), cudaMemcpyHostToDevice));
        gpu_reorder_f32(d_a, m1, m2, d_b, nullptr);
        CHECK_CUDA_ERROR(cudaGetLastError());
        std::vector<float> local(m);
        CHECK_CUDA_ERROR(cudaMemcpy(local.data(), d_b, m * sizeof(float), cudaMemcpyDeviceToHost));
        gpu_reorder_f32(d_b, m2, m1, d_a, nullptr);
        CHECK_CUDA_ERROR(cudaGetLastError());
        std::vector<float> back(m);
        CHECK_CUDA_ERROR(cudaMemcpy(back.data(), d_a, m * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaFree(d_a));
        CHECK_CUDA_ERROR(cudaFree(d_b));
        size_t bad = 0, bad_map = 0;
        for (size_t i = 0; i < m; i++)
            bad += back[i] != h[i];
        for (size_t k1 = 0; k1 < m1; k1 += 97)
            for (size_t k2 = 0; k2 < m2; k2 += 89)
                bad_map += local[k1 * m2 + k2] != h[k2 * m1 + k1];
        INFO("reorder m1={:d} m2={:d}: {:d} round-trip mismatches, {:d} mapping mismatches", m1, m2,
             bad, bad_map);
        return bad == 0 && bad_map == 0;
    }

    void main_thread() override {
        bool ok = check_reorder(4096, 2048);
        ok = check_reorder(512, 256) && ok;
        // Accuracy runs take 2^20 points per size. With perf: true the batches are large
        // enough that the working set leaves L2 and the timing lines mean something; the
        // production four-step shape is n=8192, batch 8192.
        const size_t cases[][2] = {{32, 65536},  {64, 65536},  {256, 32768},
                                   {1024, 8192}, {4096, 4096}, {8192, 8192}};
        for (const auto& c : cases)
            ok = run_case(c[0], perf ? c[1] : (1 << 20) / c[0]) && ok;
        if (ok)
            TEST_PASSED();
        else
            TEST_FAILED();
    }

private:
    const bool perf;
};

REGISTER_KOTEKAN_STAGE(testTensorFFT);
