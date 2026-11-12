// Standalone test for the incoherent-beam dedispersion kernels: runs the GPU launchers and the CPU
// references on simulated feed-summed power, compares them, and checks that injected dispersed
// pulses are found and an undispersed one is not. This tests the kernels only; it does NOT exercise
// the kotekan ring-buffer / pipeline integration. Modeled on testPLMaskFRB1.

#include "Config.hpp"                   // for Config
#include "Stage.hpp"                    // for Stage
#include "StageFactory.hpp"             // for REGISTER_KOTEKAN_STAGE
#include "bufferContainer.hpp"          // for bufferContainer
#include "cudaIncoherentDedisperse.hpp" // for launch_incoherent_*, cpu_incoherent_*
#include "cudaUtils.hpp"                // for CHECK_CUDA_ERROR
#include "cuda_runtime.h"               // for cudaMalloc, cudaMemcpy, ...
#include "errors.h"                     // for TEST_PASSED
#include "kotekanLogging.hpp"           // for FATAL_ERROR, INFO

#include <algorithm>  // for max
#include <cmath>      // for abs, llround, lround
#include <cstddef>    // for ptrdiff_t
#include <cstdint>    // for int32_t, int64_t, uint32_t, uint64_t
#include <functional> // for function
#include <random>     // for mt19937_64, normal_distribution
#include <string>     // for string
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

class testIncoherentDedisperse : public kotekan::Stage {
public:
    testIncoherentDedisperse(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container, [](const kotekan::Stage& stage) {
            return const_cast<kotekan::Stage&>(stage).main_thread();
        }) {}

    virtual ~testIncoherentDedisperse() {}

    void main_thread() override {
        using std::int32_t;
        using std::int64_t;
        using std::ptrdiff_t;
        using std::uint32_t;
        using std::uint64_t;

        // 256 frequencies from 800 to 400 MHz, sampled every 256 FPGA samples, searched at DMs
        // 0, 10, ... 630 pc/cm^3
        const ptrdiff_t num_frequencies = 256;
        const ptrdiff_t num_times = 32;
        const ptrdiff_t num_chunks = 1024;
        const ptrdiff_t num_dms = 64;
        const ptrdiff_t history_size = 16384;
        const ptrdiff_t sums_size = 64;
        const float alpha = 1.0f / 1024;
        const float clip_sigmas = 10;
        const double sample_ms = 256 * 5.12e-3;
        const double dm_step = 10;
        // Dispersion delay in ms for DM 1 pc/cm^3 between 1 GHz and infinite frequency
        const double dispersion_ms = 4.148808;

        std::vector<double> freq_ghz(num_frequencies);
        for (ptrdiff_t f = 0; f < num_frequencies; ++f)
            freq_ghz[f] = 0.8 - 0.4 * (f + 0.5) / num_frequencies;

        std::vector<int32_t> delay(num_dms * num_frequencies);
        std::vector<int32_t> max_delay(num_dms, 0);
        for (ptrdiff_t d = 0; d < num_dms; ++d) {
            for (ptrdiff_t f = 0; f < num_frequencies; ++f) {
                const double delay_ms =
                    dispersion_ms * dm_step * d
                    * (1 / (freq_ghz[f] * freq_ghz[f]) - 1 / (freq_ghz[0] * freq_ghz[0]));
                delay[f + num_frequencies * d] = int32_t(std::lround(delay_ms / sample_ms));
                max_delay[d] = std::max(max_delay[d], delay[f + num_frequencies * d]);
            }
            if (max_delay[d] + num_times > history_size)
                FATAL_ERROR("test misconfigured: delay {} of DM {} does not fit the history",
                            max_delay[d], d);
        }

        // Frequency 5 is excluded, and carries a strong signal
        const ptrdiff_t excluded_frequency = 5;
        std::vector<float> weight(num_frequencies, 1);
        weight[excluded_frequency] = 0;

        // Injected signals, in units of the noise of one sample of one frequency.
        // Pulse 1: all frequencies, one sample wide; expected significance sqrt(255) * 1 = 16.
        const ptrdiff_t dm1 = 30;
        const int64_t time1 = 20000;
        const double amp1 = 1.0;
        // Pulse 2: upper half of the frequency axis only, four samples wide; expected significance
        // sqrt(128 * 4) * 0.75 = 17, in the template of that half (11 in the full band).
        const ptrdiff_t dm2 = 50;
        const int64_t time2 = 24000;
        const int width2 = 4;
        const double amp2 = 0.75;
        // An undispersed pulse in all frequencies: 80 sigma at DM 0 if it is not subtracted
        const int64_t time_rfi = 15000;
        const double amp_rfi = 5.0;

        // Simulated power: sample count `s0`, mean power `level`, relative noise `noise`
        const uint64_t s0 = 256 * 128;
        const double level = 100;
        const double noise = 1 / std::sqrt(double(s0));
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
        std::normal_distribution<double> normal(0.0, 1.0);

        const auto signal = [&](const int64_t t, const ptrdiff_t f) {
            double s = 0;
            if (t == time1 + delay[f + num_frequencies * dm1])
                s += amp1;
            const int64_t t2 = t - time2 - delay[f + num_frequencies * dm2];
            if (f >= num_frequencies / 2 && t2 >= 0 && t2 < width2)
                s += amp2;
            if (t == time_rfi)
                s += amp_rfi;
            if (f == excluded_frequency && t % 3 == 0)
                s += 4;
            return s;
        };

        // Host state for the CPU reference
        std::vector<uint64_t> h_S012tilde(num_times * num_frequencies * 3);
        std::vector<float> h_z(num_times * num_frequencies);
        std::vector<float> h_mean(num_frequencies, 0), h_variance(num_frequencies, 0);
        std::vector<uint32_t> h_count(num_frequencies, 0);
        std::vector<float> h_history(num_frequencies * history_size, 0);
        std::vector<float> h_sums(num_dms * incoherent_dedisperse_num_subbands * sums_size, 0);
        std::vector<float> cpu_snr(num_dms), gpu_snr(num_dms);
        std::vector<int32_t> cpu_time(num_dms), cpu_width(num_dms), cpu_template(num_dms);
        std::vector<int32_t> gpu_time(num_dms), gpu_width(num_dms), gpu_template(num_dms);

        // Device state
        uint64_t* d_S012tilde = nullptr;
        float *d_z = nullptr, *d_mean = nullptr, *d_variance = nullptr, *d_history = nullptr;
        float *d_sums = nullptr, *d_weight = nullptr, *d_snr = nullptr;
        uint32_t* d_count = nullptr;
        int32_t *d_delay = nullptr, *d_max_delay = nullptr;
        int32_t *d_time = nullptr, *d_width = nullptr, *d_template = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d_S012tilde, h_S012tilde.size() * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_z, h_z.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mean, num_frequencies * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_variance, num_frequencies * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_count, num_frequencies * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_history, h_history.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_sums, h_sums.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_weight, num_frequencies * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_delay, delay.size() * sizeof(int32_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_max_delay, num_dms * sizeof(int32_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_snr, num_dms * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_time, num_dms * sizeof(int32_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_width, num_dms * sizeof(int32_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_template, num_dms * sizeof(int32_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_mean, 0, num_frequencies * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_variance, 0, num_frequencies * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_count, 0, num_frequencies * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMemset(d_history, 0, h_history.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_sums, 0, h_sums.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_weight, weight.data(), num_frequencies * sizeof(float),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_delay, delay.data(), delay.size() * sizeof(int32_t),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_max_delay, max_delay.data(), num_dms * sizeof(int32_t),
                                    cudaMemcpyHostToDevice));

        // Most significant sample of each DM over the whole run
        std::vector<float> run_snr(num_dms, 0);
        std::vector<int64_t> run_time(num_dms, 0);
        std::vector<int32_t> run_width(num_dms, 0), run_template(num_dms, 0);

        for (ptrdiff_t chunk = 0; chunk < num_chunks; ++chunk) {
            const int64_t time_begin = chunk * num_times;

            for (ptrdiff_t t = 0; t < num_times; ++t) {
                for (ptrdiff_t f = 0; f < num_frequencies; ++f) {
                    const ptrdiff_t tf = f + num_frequencies * t;
                    // A few samples have no data
                    const bool lost = (31 * (time_begin + t) + 17 * f) % 997 == 0;
                    const double x =
                        level * (1 + noise * (normal(rng) + signal(time_begin + t, f)));
                    h_S012tilde[3 * tf + 0] = lost ? 0 : s0;
                    h_S012tilde[3 * tf + 1] = lost ? 0 : uint64_t(std::llround(double(s0) * x));
                    h_S012tilde[3 * tf + 2] = 0;
                }
            }

            cpu_incoherent_normalize(h_z.data(), h_mean.data(), h_variance.data(), h_count.data(),
                                     h_S012tilde.data(), num_frequencies, num_times, alpha,
                                     clip_sigmas);
            cpu_incoherent_zero_dm(h_history.data(), h_z.data(), num_frequencies, num_times,
                                   history_size, time_begin);
            cpu_incoherent_dedisperse(
                cpu_snr.data(), cpu_time.data(), cpu_width.data(), cpu_template.data(),
                h_sums.data(), h_history.data(), delay.data(), max_delay.data(), weight.data(),
                num_dms, num_frequencies, num_times, history_size, sums_size, time_begin);

            CHECK_CUDA_ERROR(cudaMemcpy(d_S012tilde, h_S012tilde.data(),
                                        h_S012tilde.size() * sizeof(uint64_t),
                                        cudaMemcpyHostToDevice));
            launch_incoherent_normalize(d_z, d_mean, d_variance, d_count, d_S012tilde,
                                        num_frequencies, num_times, alpha, clip_sigmas,
                                        /*stream=*/0);
            launch_incoherent_zero_dm(d_history, d_z, num_frequencies, num_times, history_size,
                                      time_begin, /*stream=*/0);
            launch_incoherent_dedisperse(d_snr, d_time, d_width, d_template, d_sums, d_history,
                                         d_delay, d_max_delay, d_weight, num_dms, num_frequencies,
                                         num_times, history_size, sums_size, time_begin,
                                         /*stream=*/0);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(
                cudaMemcpy(gpu_snr.data(), d_snr, num_dms * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_time.data(), d_time, num_dms * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_width.data(), d_width, num_dms * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_template.data(), d_template, num_dms * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));

            for (ptrdiff_t d = 0; d < num_dms; ++d) {
                // The GPU and the CPU may round differently
                if (!(std::abs(gpu_snr[d] - cpu_snr[d]) <= 1e-3f * (1 + std::abs(cpu_snr[d])))
                    || gpu_time[d] != cpu_time[d] || gpu_width[d] != cpu_width[d]
                    || gpu_template[d] != cpu_template[d])
                    FATAL_ERROR("Mismatch in chunk {}, DM {}: gpu snr={} time={} width={} "
                                "template={}, cpu snr={} time={} width={} template={}",
                                chunk, d, gpu_snr[d], gpu_time[d], gpu_width[d], gpu_template[d],
                                cpu_snr[d], cpu_time[d], cpu_width[d], cpu_template[d]);
                if (gpu_snr[d] > run_snr[d]) {
                    run_snr[d] = gpu_snr[d];
                    run_time[d] = time_begin - max_delay[d] + gpu_time[d];
                    run_width[d] = gpu_width[d];
                    run_template[d] = gpu_template[d];
                }
            }
        }

        CHECK_CUDA_ERROR(cudaFree(d_S012tilde));
        CHECK_CUDA_ERROR(cudaFree(d_z));
        CHECK_CUDA_ERROR(cudaFree(d_mean));
        CHECK_CUDA_ERROR(cudaFree(d_variance));
        CHECK_CUDA_ERROR(cudaFree(d_count));
        CHECK_CUDA_ERROR(cudaFree(d_history));
        CHECK_CUDA_ERROR(cudaFree(d_sums));
        CHECK_CUDA_ERROR(cudaFree(d_weight));
        CHECK_CUDA_ERROR(cudaFree(d_delay));
        CHECK_CUDA_ERROR(cudaFree(d_max_delay));
        CHECK_CUDA_ERROR(cudaFree(d_snr));
        CHECK_CUDA_ERROR(cudaFree(d_time));
        CHECK_CUDA_ERROR(cudaFree(d_width));
        CHECK_CUDA_ERROR(cudaFree(d_template));

        INFO("Pulse 1: DM {} snr={} time={} width={} template={}", dm1, run_snr[dm1], run_time[dm1],
             run_width[dm1], run_template[dm1]);
        if (!(run_snr[dm1] > 13 && run_snr[dm1] < 19) || run_time[dm1] != time1
            || run_width[dm1] != 1 || run_template[dm1] != 0)
            FATAL_ERROR("Pulse 1 not found as injected: expected snr 16, time {}, width 1, "
                        "template 0",
                        time1);

        INFO("Pulse 2: DM {} snr={} time={} width={} template={}", dm2, run_snr[dm2], run_time[dm2],
             run_width[dm2], run_template[dm2]);
        if (!(run_snr[dm2] > 14 && run_snr[dm2] < 20) || run_time[dm2] != time2 + width2 - 1
            || run_width[dm2] != width2 || run_template[dm2] != 2)
            FATAL_ERROR("Pulse 2 not found as injected: expected snr 17, time {}, width {}, "
                        "template 2",
                        time2 + width2 - 1, width2);

        // Away from the injected DMs only noise remains: the undispersed pulse and the excluded
        // frequency must not show
        float noise_snr = 0;
        for (ptrdiff_t d = 0; d < num_dms; ++d)
            if (std::abs(d - dm1) > 3 && std::abs(d - dm2) > 3)
                noise_snr = std::max(noise_snr, run_snr[d]);
        INFO("DM 0 snr={}; largest snr away from the pulses={}", run_snr[0], noise_snr);
        if (noise_snr > 8)
            FATAL_ERROR("Significance {} away from the injected pulses", noise_snr);

        INFO("testIncoherentDedisperse: GPU and CPU matched, injected pulses found.");
        TEST_PASSED();
    }
};

REGISTER_KOTEKAN_STAGE(testIncoherentDedisperse);
