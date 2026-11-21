// Self-checking test for the PFB kernels: a 4-tap windowed-sinc fold (pfbPrototype.hpp,
// hamming window, the fftwEngine default) ahead of the window FFT. Tone + noise windows run
// through the GPU fold and FFT and are checked against a double-precision CPU fold + DFT,
// and a half-bin tone checks the leakage improvement over the plain FFT. The device reads
// the prototype as __half; the CPU reference folds the same half-rounded weights, so the
// checks measure the GPU arithmetic while the leakage runs measure the rounded filter
// itself. The fused fold kernel is also checked alone against a CPU fold, at a shape for
// each tile width it uses. With perf: true it also times the fold kernels at production
// sizes.

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp" // for bufferContainer
#include "crsPacket.hpp"       // for pack12
#include "cudaUtils.hpp"       // for CHECK_CUDA_ERROR
#include "errors.h"            // for TEST_PASSED, TEST_FAILED
#include "fEngineKernels.hpp"  // for gpu_untangle_quantize, gpu_unpack12_fp16_t
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "pfbKernels.hpp"      // for gpu_pfb_fold, gpu_pfb_unpack12_fp16_t
#include "pfbPrototype.hpp"    // for pfb_prototype
#include "tensorFFT.hpp"       // for gpu_tensor_fft

#include <algorithm>   // for fill, max, min
#include <cmath>       // for cos, sqrt, log10, lround, nearbyint, fma, frexp, ldexp
#include <complex>     // for complex, polar, conj, norm
#include <cstring>     // for memset
#include <cuda_fp16.h> // for __half2, __low2float, __high2float
#include <cufft.h>     // for cufftPlan1d, cufftExecC2C
#include <random>      // for mt19937, normal_distribution
#include <stdint.h>    // for uint8_t, int16_t, uint64_t, uintptr_t
#include <string>      // for string
#include <sys/mman.h>  // for mmap, munmap, MAP_HUGETLB
#include <vector>      // for vector

namespace {

constexpr int TAPS = 4;

// The prototype as the device reads it: rounded once to __half, widened back to float so
// the CPU reference folds exactly the weights the GPU folds.
std::vector<float> round_to_half(const std::vector<float>& proto) {
    std::vector<float> h(proto.size());
    for (size_t i = 0; i < proto.size(); i++)
        h[i] = __half2float(__float2half_rn(proto[i]));
    return h;
}

// Raw 12-bit samples with continuous tone phase across windows: 2048 * amp *
// cos(2 pi tone_bins g / n) + gaussian noise, clamped to -2048..2047.
std::vector<int16_t> make_signal(size_t windows, size_t n, double tone_bins, double amp,
                                 double noise_rms, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_rms);
    std::vector<int16_t> raw(windows * n);
    for (size_t g = 0; g < raw.size(); g++) {
        double v = 2048.0 * amp * std::cos(2.0 * M_PI * tone_bins * (double)g / (double)n);
        if (noise_rms > 0)
            v += noise(rng);
        const long r = std::lround(v);
        raw[g] = (int16_t)(r < -2048 ? -2048 : (r > 2047 ? 2047 : r));
    }
    return raw;
}

// Direct real DFT under the fftWindowJob.hpp convention: X[k] = (2/n) sum_j y[j] W^(jk),
// k = 0 .. n/2 - 1, twiddles from a table in double. n must be a power of two.
std::vector<std::complex<double>> real_dft(const std::vector<double>& y, size_t n) {
    std::vector<std::complex<double>> tab(n);
    for (size_t j = 0; j < n; j++)
        tab[j] = std::polar(1.0, -2.0 * M_PI * (double)j / (double)n);
    std::vector<std::complex<double>> x(n / 2);
    for (size_t k = 0; k < n / 2; k++) {
        std::complex<double> acc = 0;
        for (size_t j = 0; j < n; j++)
            acc += y[j] * tab[(j * k) & (n - 1)];
        x[k] = (2.0 / (double)n) * acc;
    }
    return x;
}

// The window ending the 4-tap fold at output window s, unpacked to raw / 2048.
std::vector<double> cpu_fold(const std::vector<int16_t>& raw, size_t s, size_t n,
                             const std::vector<float>& proto) {
    std::vector<double> y(n);
    for (size_t p = 0; p < n; p++) {
        double acc = 0;
        for (int q = 0; q < TAPS; q++)
            acc += (double)proto[q * n + p] * ((double)raw[(s - 3 + q) * n + p] / 2048.0);
        y[p] = acc;
    }
    return y;
}

// Untangle the m-point C2C FFT of the real-as-complex window into the real spectrum, in
// double: the math of gpu_untangle_quantize with the 2/n scale, bins 0 .. m-1.
std::vector<std::complex<double>> untangle(const std::vector<std::complex<double>>& z, size_t n) {
    const size_t m = n / 2;
    std::vector<std::complex<double>> x(m);
    for (size_t k = 0; k < m; k++) {
        const std::complex<double> zk = z[k];
        const std::complex<double> zmk = std::conj(z[k == 0 ? 0 : m - k]);
        const std::complex<double> ze = 0.5 * (zk + zmk);
        const std::complex<double> zo = std::complex<double>(0.0, -0.5) * (zk - zmk);
        const std::complex<double> w = std::polar(1.0, -2.0 * M_PI * (double)k / (double)n);
        x[k] = (2.0 / (double)n) * (ze + w * zo);
    }
    return x;
}

double rel_l2(const std::vector<std::complex<double>>& a,
              const std::vector<std::complex<double>>& b) {
    double num = 0, den = 0;
    for (size_t k = 0; k < a.size(); k++) {
        num += std::norm(a[k] - b[k]);
        den += std::norm(b[k]);
    }
    return std::sqrt(num / den);
}

// The 4+4 offset encoding of fftWindowJob.hpp, in double: round half to even like
// __float2int_rn, clamp to +-7, +8, real in the high nibble.
uint8_t encode_ref(std::complex<double> x, double gain) {
    const auto q = [&](double v) {
        const int r = (int)std::nearbyint(gain * v);
        return (r < -7 ? -7 : (r > 7 ? 7 : r)) + 8;
    };
    return (uint8_t)((q(x.real()) << 4) | q(x.imag()));
}

// Count bytes whose nibbles differ from the reference by more than one count.
int count_bad_bytes(const std::vector<uint8_t>& got, const std::vector<std::complex<double>>& x,
                    double gain) {
    int bad = 0;
    for (size_t k = 0; k < got.size(); k++) {
        const uint8_t ref = encode_ref(x[k], gain);
        if (std::abs((got[k] >> 4) - (ref >> 4)) > 1 || std::abs((got[k] & 15) - (ref & 15)) > 1)
            bad++;
    }
    return bad;
}

// Worst bin two or more channels from the tone, dB relative to the peak bin.
double leakage_db(const std::vector<std::complex<double>>& x, double tone_bins) {
    double peak = 0, leak = 0;
    for (size_t k = 0; k < x.size(); k++) {
        const double a = std::abs(x[k]);
        peak = peak > a ? peak : a;
        if (std::abs((double)k - tone_bins) >= 2.0 && a > leak)
            leak = a;
    }
    return 20.0 * std::log10(leak / peak);
}

// The spacing of fp16 values at x.
float half_ulp(float x) {
    int e;
    std::frexp(std::max(std::abs(x), 6.103515625e-05f), &e);
    return std::ldexp(1.0f, e - 11);
}

// The four-step shape of an n-point window: m = n / 2 = m1 * m2, m1 >= m2.
void fourstep_shape(size_t n, size_t& m1, size_t& m2) {
    int log2m = 0;
    while ((1ull << log2m) < n / 2)
        log2m++;
    m1 = 1ull << ((log2m + 1) / 2);
    m2 = n / 2 / m1;
}

} // namespace

class testPfb : public kotekan::Stage {
public:
    testPfb(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        perf(config.get_default<bool>(unique_name, "perf", false)) {}

    ~testPfb() override = default;

    // Upload a half-exact prototype as a full __half table: the fp32 fold reads all of it,
    // the fused kernel only its symmetric first half.
    __half* upload_proto(const std::vector<float>& proto) {
        std::vector<__half> h(proto.size());
        for (size_t i = 0; i < proto.size(); i++)
            h[i] = __float2half_rn(proto[i]);
        __half* d;
        CHECK_CUDA_ERROR(cudaMalloc(&d, h.size() * sizeof(__half)));
        CHECK_CUDA_ERROR(
            cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice));
        return d;
    }

    // The untangle twiddle tables of cudaSparkFEngine: W^k = w_hi[k >> 12] * w_lo[k & 4095].
    void upload_twiddles(size_t n, float2** d_whi, float2** d_wlo) {
        const size_t hi_entries = (n + 4095) / 4096;
        std::vector<float2> whi(hi_entries), wlo(4096);
        const double step = -2.0 * M_PI / (double)n;
        for (size_t j = 0; j < hi_entries; j++) {
            const std::complex<double> w = std::polar(1.0, step * (double)(j * 4096));
            whi[j] = make_float2((float)w.real(), (float)w.imag());
        }
        for (size_t j = 0; j < 4096; j++) {
            const std::complex<double> w = std::polar(1.0, step * (double)j);
            wlo[j] = make_float2((float)w.real(), (float)w.imag());
        }
        CHECK_CUDA_ERROR(cudaMalloc(d_whi, hi_entries * sizeof(float2)));
        CHECK_CUDA_ERROR(cudaMalloc(d_wlo, 4096 * sizeof(float2)));
        CHECK_CUDA_ERROR(
            cudaMemcpy(*d_whi, whi.data(), hi_entries * sizeof(float2), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(
            cudaMemcpy(*d_wlo, wlo.data(), 4096 * sizeof(float2), cudaMemcpyHostToDevice));
    }

    // fp32 path: fold kernel + cuFFT + the existing untangle/quantize, against the CPU
    // reference. Returns the worst rel L2 through output argument for logging.
    bool run_fp32(size_t n, const std::vector<int16_t>& raw, size_t windows,
                  const std::vector<float>& proto, double gain,
                  std::vector<std::vector<std::complex<double>>>& x_gpu) {
        const size_t m = n / 2;

        std::vector<float> xf(raw.size());
        for (size_t i = 0; i < raw.size(); i++)
            xf[i] = (float)(raw[i] * (1.0 / 2048.0));

        float *d_x, *d_fold;
        float2 *d_spec, *d_whi, *d_wlo;
        uint8_t* d_out;
        CHECK_CUDA_ERROR(cudaMalloc(&d_x, xf.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_fold, n * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_spec, m * sizeof(float2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_out, m));
        CHECK_CUDA_ERROR(
            cudaMemcpy(d_x, xf.data(), xf.size() * sizeof(float), cudaMemcpyHostToDevice));
        __half* d_proto = upload_proto(proto);
        upload_twiddles(n, &d_whi, &d_wlo);
        cufftHandle plan;
        if (cufftPlan1d(&plan, (int)m, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
            ERROR("cuFFT plan failed at m={:d}", m);
            return false;
        }

        bool ok = true;
        double worst = 0;
        int bad_bytes = 0;
        x_gpu.clear();
        for (size_t s = TAPS - 1; s < windows; s++) {
            gpu_pfb_fold(d_x + (s - 3) * n, d_x + (s - 2) * n, d_x + (s - 1) * n, d_x + s * n,
                         d_proto, n, d_fold, nullptr);
            if (cufftExecC2C(plan, (cufftComplex*)d_fold, (cufftComplex*)d_spec, CUFFT_FORWARD)
                != CUFFT_SUCCESS) {
                ERROR("cuFFT exec failed");
                ok = false;
                break;
            }
            gpu_untangle_quantize(d_spec, m, d_whi, d_wlo, (float)(2.0 / (double)n), (float)gain,
                                  nullptr, nullptr, d_out, nullptr);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaGetLastError());

            std::vector<float2> h_spec(m);
            std::vector<uint8_t> h_out(m);
            CHECK_CUDA_ERROR(
                cudaMemcpy(h_spec.data(), d_spec, m * sizeof(float2), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, m, cudaMemcpyDeviceToHost));

            std::vector<std::complex<double>> z(m);
            for (size_t k = 0; k < m; k++)
                z[k] = std::complex<double>(h_spec[k].x, h_spec[k].y);
            const auto x = untangle(z, n);
            const auto x_ref = real_dft(cpu_fold(raw, s, n, proto), n);
            const double err = rel_l2(x, x_ref);
            worst = worst > err ? worst : err;
            bad_bytes += count_bad_bytes(h_out, x_ref, gain);
            x_gpu.push_back(x);
        }

        cufftDestroy(plan);
        CHECK_CUDA_ERROR(cudaFree(d_x));
        CHECK_CUDA_ERROR(cudaFree(d_proto));
        CHECK_CUDA_ERROR(cudaFree(d_fold));
        CHECK_CUDA_ERROR(cudaFree(d_spec));
        CHECK_CUDA_ERROR(cudaFree(d_out));
        CHECK_CUDA_ERROR(cudaFree(d_whi));
        CHECK_CUDA_ERROR(cudaFree(d_wlo));

        INFO("fp32 fold+cuFFT n={:d}: rel L2 vs double {:.3g} (limit 1e-5), quantize bytes off "
             "by >1: {:d}",
             n, worst, bad_bytes);
        return ok && worst < 1e-5 && bad_bytes == 0;
    }

    // fp16 four-step path with the tensor-core FFT, from 12-bit packed payload slots read
    // through lookup tables as in production: the fused fold + unpack kernel when fold is
    // set, the plain unpack otherwise, then the existing twiddle/transpose and quantize.
    // The fold runs in fp32 inside the kernel; the tensor cores are in the FFTs.
    bool run_fp16(size_t n, size_t sp, const std::vector<int16_t>& raw, size_t windows,
                  const std::vector<float>& proto, double gain, bool fold, bool zero_history,
                  std::vector<std::vector<std::complex<double>>>& x_gpu) {
        const size_t m = n / 2;
        size_t m1, m2;
        fourstep_shape(n, m1, m2);
        const size_t num_packets = n / sp;
        const size_t slot = 3 * sp / 2;

        // Pack the raw samples into 12-bit payload slots and point per-window lookup
        // tables at them.
        std::vector<uint8_t> pk(raw.size() / sp * slot);
        for (size_t i = 0; i < raw.size(); i++)
            pack12(pk.data() + (i / sp) * slot, i % sp, raw[i]);
        uint8_t* d_pk;
        CHECK_CUDA_ERROR(cudaMalloc(&d_pk, pk.size()));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pk, pk.data(), pk.size(), cudaMemcpyHostToDevice));
        std::vector<uint64_t> luts(windows * num_packets);
        for (size_t p = 0; p < luts.size(); p++)
            luts[p] = (uint64_t)(uintptr_t)(d_pk + p * slot);
        uint64_t* d_luts;
        CHECK_CUDA_ERROR(cudaMalloc(&d_luts, luts.size() * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_luts, luts.data(), luts.size() * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice));

        __half* d_proto = upload_proto(proto);
        __half2 *d_a, *d_b;
        uint8_t* d_out;
        float2 *d_whi, *d_wlo;
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, m * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, m * sizeof(__half2)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_out, m));
        upload_twiddles(n, &d_whi, &d_wlo);

        // The twiddle pass divides by m1 to stay in fp16 range; fold it back into the scale.
        const float scale = (float)(2.0 / (double)n * (double)m1);

        double worst = 0;
        int bad_bytes = 0;
        x_gpu.clear();
        for (size_t s = TAPS - 1; s < windows; s++) {
            if (fold) {
                pfb_taps taps;
                for (int q = 0; q < TAPS; q++)
                    taps.lut[q] = (zero_history && q < TAPS - 1)
                                      ? nullptr
                                      : d_luts + (s - 3 + q) * num_packets;
                gpu_pfb_unpack12_fp16_t(taps, (int)sp, m1, m2, d_proto, d_a, nullptr);
            } else {
                gpu_unpack12_fp16_t(d_luts + s * num_packets, (int)sp, m1, m2, d_a, nullptr);
            }
            gpu_tensor_fft(d_a, d_b, m1, m2, nullptr);
            gpu_twiddle_transpose(d_b, m1, m2, d_whi, d_wlo, 1.0f / m1, tensor_fft_split(m1), d_a,
                                  nullptr);
            gpu_tensor_fft(d_a, d_b, m2, m1, nullptr);
            gpu_quantize_fourstep(d_b, m1, m2, d_whi, d_wlo, scale, (float)gain, nullptr, nullptr,
                                  tensor_fft_split(m2), d_out, nullptr);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaGetLastError());

            std::vector<__half2> h_b(m);
            std::vector<uint8_t> h_out(m);
            CHECK_CUDA_ERROR(
                cudaMemcpy(h_b.data(), d_b, m * sizeof(__half2), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, m, cudaMemcpyDeviceToHost));

            // spec[k1][k2] holds Z[k2 * m1 + k1] / m1 (fourstep layout, twiddle scaling).
            std::vector<std::complex<double>> z(m);
            for (size_t k1 = 0; k1 < m1; k1++)
                for (size_t k2 = 0; k2 < m2; k2++)
                    z[k2 * m1 + k1] = (double)m1
                                      * std::complex<double>(__low2float(h_b[k1 * m2 + k2]),
                                                             __high2float(h_b[k1 * m2 + k2]));
            const auto x = untangle(z, n);

            std::vector<std::complex<double>> x_ref;
            if (fold) {
                auto hist = raw;
                if (zero_history)
                    std::fill(hist.begin() + (s - 3) * n, hist.begin() + s * n, (int16_t)0);
                x_ref = real_dft(cpu_fold(hist, s, n, proto), n);
            } else {
                std::vector<double> newest(n);
                for (size_t p = 0; p < n; p++)
                    newest[p] = (double)raw[s * n + p] / 2048.0;
                x_ref = real_dft(newest, n);
            }
            const double err = rel_l2(x, x_ref);
            worst = worst > err ? worst : err;
            bad_bytes += count_bad_bytes(h_out, x_ref, gain);
            x_gpu.push_back(x);
        }

        CHECK_CUDA_ERROR(cudaFree(d_pk));
        CHECK_CUDA_ERROR(cudaFree(d_luts));
        CHECK_CUDA_ERROR(cudaFree(d_proto));
        CHECK_CUDA_ERROR(cudaFree(d_a));
        CHECK_CUDA_ERROR(cudaFree(d_b));
        CHECK_CUDA_ERROR(cudaFree(d_out));
        CHECK_CUDA_ERROR(cudaFree(d_whi));
        CHECK_CUDA_ERROR(cudaFree(d_wlo));

        INFO("fp16 tensor four-step {:s}n={:d}: rel L2 vs double {:.4f}% (limit 0.5%), "
             "quantize bytes off by >1: {:d}",
             fold ? (zero_history ? "pfb (zero history) " : "pfb ") : "plain ", n, 100 * worst,
             bad_bytes);
        return worst < 0.005 && bad_bytes == 0;
    }

    // The fused fold kernel alone against a CPU fold of the same half-rounded weights, fp32
    // multiply-adds in tap order and one rounding to fp16, with full history and with null
    // history tables. The kernel picks its tile width from n and sp, so main_thread runs a
    // shape for each width.
    bool run_fold_exact(size_t n, size_t sp) {
        const size_t m = n / 2, num_packets = n / sp, slot = 3 * sp / 2;
        size_t m1, m2;
        fourstep_shape(n, m1, m2);
        const auto raw = make_signal(TAPS, n, 333.25, 1500.0 / 2048.0, 100.0, 2468);
        const auto proto = round_to_half(pfb_prototype((int)n, TAPS));

        std::vector<uint8_t> pk(TAPS * num_packets * slot);
        for (size_t i = 0; i < raw.size(); i++)
            pack12(pk.data() + (i / sp) * slot, i % sp, raw[i]);
        std::vector<uint64_t> luts(TAPS * num_packets);
        uint8_t* d_pk;
        uint64_t* d_luts;
        __half2* d_out;
        CHECK_CUDA_ERROR(cudaMalloc(&d_pk, pk.size()));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pk, pk.data(), pk.size(), cudaMemcpyHostToDevice));
        for (size_t p = 0; p < luts.size(); p++)
            luts[p] = (uint64_t)(uintptr_t)(d_pk + p * slot);
        CHECK_CUDA_ERROR(cudaMalloc(&d_luts, luts.size() * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_luts, luts.data(), luts.size() * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMalloc(&d_out, m * sizeof(__half2)));
        __half* d_proto = upload_proto(proto);

        size_t bad = 0;
        std::vector<__half2> h_out(m);
        for (int first : {0, TAPS - 1}) {
            pfb_taps taps;
            for (int q = 0; q < TAPS; q++)
                taps.lut[q] = q < first ? nullptr : d_luts + q * num_packets;
            gpu_pfb_unpack12_fp16_t(taps, (int)sp, m1, m2, d_proto, d_out, nullptr);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(
                cudaMemcpy(h_out.data(), d_out, m * sizeof(__half2), cudaMemcpyDeviceToHost));
            for (size_t p = 0; p < n; p++) {
                float acc = 0.0f;
                for (int q = first; q < TAPS; q++)
                    acc = std::fma(proto[q * n + p], raw[q * n + p] * (1.0f / 2048.0f), acc);
                const float want = __half2float(__float2half_rn(acc));
                // Complex sample p / 2 = a * m2 + b is out[b * m1 + a].
                const __half2 g = h_out[(p / 2 % m2) * m1 + p / 2 / m2];
                const float got = p % 2 ? __high2float(g) : __low2float(g);
                if (std::abs(got - want) > half_ulp(want))
                    bad++;
            }
        }

        CHECK_CUDA_ERROR(cudaFree(d_pk));
        CHECK_CUDA_ERROR(cudaFree(d_luts));
        CHECK_CUDA_ERROR(cudaFree(d_out));
        CHECK_CUDA_ERROR(cudaFree(d_proto));
        INFO("fused fold n={:d}, {:d}-sample packets: {:d} of {:d} samples more than one fp16 "
             "ulp from the CPU fold",
             n, sp, bad, 2 * n);
        return bad == 0;
    }

    // The leakage comparison: a tone halfway between bins, no noise; the PFB must leak at
    // least 30 dB less than the plain FFT into bins two or more channels away.
    bool run_leakage_fp32(size_t n, double tone_bins) {
        const auto raw = make_signal(TAPS, n, tone_bins, 1800.0 / 2048.0, 0.0, 999);
        const auto proto = round_to_half(pfb_prototype((int)n, TAPS));

        std::vector<std::vector<std::complex<double>>> x_pfb;
        if (!run_fp32(n, raw, TAPS, proto, 1.0, x_pfb))
            return false;

        // Plain FFT of the newest window for the baseline, from the same CPU machinery
        // (the fp32 path is already verified above).
        std::vector<double> newest(n);
        for (size_t p = 0; p < n; p++)
            newest[p] = (double)raw[(TAPS - 1) * n + p] / 2048.0;
        const auto x_fft = real_dft(newest, n);

        const double fft_db = leakage_db(x_fft, tone_bins);
        const double pfb_db = leakage_db(x_pfb[0], tone_bins);
        INFO("leakage into bins >= 2 channels from a half-bin tone: plain FFT {:.1f} dB, "
             "4-tap PFB {:.1f} dB ({:.1f} dB better)",
             fft_db, pfb_db, fft_db - pfb_db);
        if (pfb_db > fft_db - 30.0) {
            ERROR("PFB leakage not at least 30 dB below the plain FFT");
            return false;
        }
        return true;
    }

    // The same leakage comparison on the fp16 tensor path, PFB against the plain unpack.
    bool run_leakage_fp16(size_t n, size_t sp, double tone_bins) {
        const auto raw = make_signal(TAPS, n, tone_bins, 1800.0 / 2048.0, 0.0, 999);
        const auto proto = round_to_half(pfb_prototype((int)n, TAPS));

        std::vector<std::vector<std::complex<double>>> x_pfb, x_fft;
        if (!run_fp16(n, sp, raw, TAPS, proto, 1.0, true, false, x_pfb)
            || !run_fp16(n, sp, raw, TAPS, proto, 1.0, false, false, x_fft))
            return false;

        const double fft_db = leakage_db(x_fft[0], tone_bins);
        const double pfb_db = leakage_db(x_pfb[0], tone_bins);
        INFO("fp16 leakage into bins >= 2 channels from a half-bin tone: plain FFT {:.1f} dB, "
             "4-tap PFB {:.1f} dB ({:.1f} dB better)",
             fft_db, pfb_db, fft_db - pfb_db);
        if (pfb_db > fft_db - 30.0) {
            ERROR("fp16 PFB leakage not at least 30 dB below the plain FFT");
            return false;
        }
        return true;
    }

    // Time the fold kernel at production window sizes (zeroed inputs; bandwidth is
    // 4 fp32 windows + the __half prototype in, one fp32 window out).
    void run_perf() {
        for (int lg : {24, 27}) {
            const size_t n = 1ull << lg;
            float *d_x, *d_fold;
            __half* d_proto;
            CHECK_CUDA_ERROR(cudaMalloc(&d_x, TAPS * n * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_proto, TAPS * n * sizeof(__half)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_fold, n * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_x, 0, TAPS * n * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_proto, 0, TAPS * n * sizeof(__half)));

            cudaEvent_t t0, t1;
            CHECK_CUDA_ERROR(cudaEventCreate(&t0));
            CHECK_CUDA_ERROR(cudaEventCreate(&t1));
            const int iters = 20;
            gpu_pfb_fold(d_x, d_x + n, d_x + 2 * n, d_x + 3 * n, d_proto, n, d_fold, nullptr);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaEventRecord(t0));
            for (int i = 0; i < iters; i++)
                gpu_pfb_fold(d_x, d_x + n, d_x + 2 * n, d_x + 3 * n, d_proto, n, d_fold, nullptr);
            CHECK_CUDA_ERROR(cudaEventRecord(t1));
            CHECK_CUDA_ERROR(cudaEventSynchronize(t1));
            float ms = 0;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, t0, t1));
            ms /= iters;
            const double gb =
                (double)n * (5.0 * sizeof(float) + (double)TAPS * sizeof(__half)) / 1e9;
            INFO("fold n=2^{:d}: {:.3f} ms, {:.1f} GB/s", lg, ms, gb / (ms / 1e3));
            CHECK_CUDA_ERROR(cudaEventDestroy(t0));
            CHECK_CUDA_ERROR(cudaEventDestroy(t1));
            CHECK_CUDA_ERROR(cudaFree(d_x));
            CHECK_CUDA_ERROR(cudaFree(d_proto));
            CHECK_CUDA_ERROR(cudaFree(d_fold));
        }
    }

    // Time the fused fold at production window sizes with 4096-sample packets (zeroed
    // inputs), the four windows' packets in device memory and in 2 MB huge pages of host
    // memory, where the payload ring keeps them. Host timing is skipped without free pages.
    void run_perf_fused() {
        const size_t sp = 4096, slot = 3 * sp / 2;
        for (int lg : {24, 26, 27}) {
            const size_t n = 1ull << lg, num_packets = n / sp, bytes = TAPS * num_packets * slot;
            const size_t len = (bytes + (2ull << 20) - 1) & ~((2ull << 20) - 1);
            size_t m1, m2;
            fourstep_shape(n, m1, m2);
            __half* d_proto;
            __half2* d_out;
            uint64_t* d_luts;
            CHECK_CUDA_ERROR(cudaMalloc(&d_proto, 2 * n * sizeof(__half)));
            CHECK_CUDA_ERROR(cudaMemset(d_proto, 0, 2 * n * sizeof(__half)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_out, n / 2 * sizeof(__half2)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_luts, TAPS * num_packets * sizeof(uint64_t)));
            for (bool host : {false, true}) {
                uint8_t* pk;
                if (host) {
                    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
                    if (p == MAP_FAILED) {
                        INFO("fused fold n=2^{:d}: no {:d} MiB of free huge pages, host packets "
                             "not timed",
                             lg, len >> 20);
                        continue;
                    }
                    pk = (uint8_t*)p;
                    memset(pk, 0, len);
                } else {
                    CHECK_CUDA_ERROR(cudaMalloc(&pk, bytes));
                    CHECK_CUDA_ERROR(cudaMemset(pk, 0, bytes));
                }
                std::vector<uint64_t> luts(TAPS * num_packets);
                for (size_t p = 0; p < luts.size(); p++)
                    luts[p] = (uint64_t)(uintptr_t)(pk + p * slot);
                CHECK_CUDA_ERROR(cudaMemcpy(d_luts, luts.data(), luts.size() * sizeof(uint64_t),
                                            cudaMemcpyHostToDevice));
                pfb_taps taps;
                for (int q = 0; q < TAPS; q++)
                    taps.lut[q] = d_luts + q * num_packets;

                cudaEvent_t t0, t1;
                CHECK_CUDA_ERROR(cudaEventCreate(&t0));
                CHECK_CUDA_ERROR(cudaEventCreate(&t1));
                const int iters = 10;
                gpu_pfb_unpack12_fp16_t(taps, (int)sp, m1, m2, d_proto, d_out, nullptr);
                CHECK_CUDA_ERROR(cudaEventRecord(t0));
                for (int i = 0; i < iters; i++)
                    gpu_pfb_unpack12_fp16_t(taps, (int)sp, m1, m2, d_proto, d_out, nullptr);
                CHECK_CUDA_ERROR(cudaEventRecord(t1));
                CHECK_CUDA_ERROR(cudaEventSynchronize(t1));
                float ms = 0;
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, t0, t1));
                INFO("fused fold n=2^{:d}, packets in {:s} memory: {:.3f} ms", lg,
                     host ? "host" : "device", ms / iters);
                CHECK_CUDA_ERROR(cudaEventDestroy(t0));
                CHECK_CUDA_ERROR(cudaEventDestroy(t1));
                if (host)
                    munmap(pk, len);
                else
                    CHECK_CUDA_ERROR(cudaFree(pk));
            }
            CHECK_CUDA_ERROR(cudaFree(d_proto));
            CHECK_CUDA_ERROR(cudaFree(d_out));
            CHECK_CUDA_ERROR(cudaFree(d_luts));
        }
    }

    // The stage's GPU-built table (pfb_proto_shared) against the first half of
    // pfb_prototype() rounded to __half.
    bool check_proto_table(size_t n) {
        const auto ref = round_to_half(pfb_prototype((int)n, TAPS));
        bool reused;
        const __half* d_table = pfb_proto_shared(0, n, TAPS, reused);
        std::vector<__half> table(ref.size() / 2);
        CHECK_CUDA_ERROR(cudaMemcpy(table.data(), d_table, table.size() * sizeof(__half),
                                    cudaMemcpyDeviceToHost));
        size_t off = 0;
        float worst = 0.0f;
        for (size_t i = 0; i < table.size(); i++) {
            const float d = std::abs(__half2float(table[i]) - ref[i]);
            off += d != 0.0f;
            worst = std::max(worst, d);
        }
        INFO("prototype table n={:d}: {:d} of {:d} weights differ from pfb_prototype(), worst by "
             "{:g}",
             n, off, table.size(), worst);
        return off == 0;
    }

    void main_thread() override {
        const size_t n = 8192, sp = 512, windows = 8;
        const auto raw = make_signal(windows, n, 777.25, 1500.0 / 2048.0, 100.0, 54321);
        const auto proto = round_to_half(pfb_prototype((int)n, TAPS));

        std::vector<std::vector<std::complex<double>>> x_gpu;
        bool ok = check_proto_table(n);
        ok = check_proto_table(1 << 20) && ok;
        ok = run_fp32(n, raw, windows, proto, 6.0, x_gpu) && ok;
        ok = run_leakage_fp32(n, 700.5) && ok;
        ok = run_fp16(n, sp, raw, windows, proto, 6.0, true, false, x_gpu) && ok;
        ok = run_fp16(n, sp, raw, TAPS, proto, 6.0, true, true, x_gpu) && ok;
        ok = run_leakage_fp16(n, sp, 700.5) && ok;
        // 128-, 64- and 32-column tiles, and a window of a single tile.
        ok = run_fold_exact(1 << 20, 4096) && ok;
        ok = run_fold_exact(8192, 512) && ok;
        ok = run_fold_exact(4096, 128) && ok;
        ok = run_fold_exact(2048, 64) && ok;
        if (perf) {
            run_perf();
            run_perf_fused();
        }
        if (ok)
            TEST_PASSED();
        else
            TEST_FAILED();
    }

private:
    const bool perf;
};

REGISTER_KOTEKAN_STAGE(testPfb);
