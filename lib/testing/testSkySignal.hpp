#ifndef TEST_SKY_SIGNAL_HPP
#define TEST_SKY_SIGNAL_HPP

#include "Config.hpp"         // for Config
#include "DataType.hpp"       // for int4x2_swapped_withoffset_t
#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

#include "json.hpp" // for json, basic_json<>::object_t, basic_json<>::value_type

#include <algorithm> // for clamp
#include <cmath>     // for log, lrintf, sqrt, M_PI
#include <complex>   // for complex, polar
#include <cstddef>   // for ptrdiff_t, size_t
#include <cstdint>   // for int64_t, uint64_t
#include <string>    // for string
#include <utility>   // for pair
#include <vector>    // for vector

/**
 * @brief Point-source tones plus noise as channelized F-engine voltages, for test generators.
 *
 * A source with amplitude `A`, direction `n = (n_x, n_y)` in the telescope grid frame (the
 * convention of the `setBBBeams` beam positions) and `k` tone cycles per frame contributes
 *
 *     E[d,p](f,t) = A w_p exp(2πi k (t - t0) / T) exp(-2πi ν_f n·r_d / c)
 *
 * to dish `d`, polarization `p`, coarse channel `f` and FPGA sample `t`. Here `t0` is
 * `first_fpga_seq_num`, `T` is `num_times`, `ν_f = (channel + k/T) Δν` is the tone frequency,
 * and `r_d` is the dish position, computed from `telescope.dish_inputs` (not via `Telescope`).
 * This is the convention of RadioTelescopeFEngine.jl, for which the `calcBBPhase` beam with
 * position `n_b` peaks at `n = n_b`. Dishes not listed in `dish_inputs` receive nothing.
 *
 * Noise is complex Gaussian with standard deviation `noise_rms` per component,
 * independent per sample and input, and repeats every `noise_period_frames` frames. Values are
 * rounded and clamped to [-7, +7]; -8 is reserved as the no-data value.
 *
 * @conf sources              List of objects, each with `amplitude` (LSB), `direction`
 *                            ([n_x, n_y]), `tone_cycles_per_frame` (integer), optional `pol`
 *                            ([w_0, w_1], default [1, 1]) and optional `window` ([begin, end),
 *                            FPGA samples; default: always on)
 * @conf noise_rms            Float. Default 0.
 * @conf seed                 Int. Noise seed. Default 1.
 * @conf noise_period_frames  Int. Default 4.
 * @conf active_inputs        List of [dish, polarization]. If set, all other inputs are zero.
 * @conf first_fpga_seq_num   Int. Default 0.
 */
class testSkySignal {
public:
    struct Source {
        float amplitude;
        double n_x, n_y;
        int tone_cycles_per_frame;
        float pol[2];
        bool windowed;
        std::int64_t window_begin, window_end;
    };

    testSkySignal(const kotekan::Config& config, const std::string& unique_name,
                  const int num_times, const std::vector<int>& frequency_channels,
                  const int num_polarizations, const int num_dishes) :
        num_times(num_times), num_frequencies(frequency_channels.size()),
        num_polarizations(num_polarizations), num_dishes(num_dishes),
        num_elements(std::ptrdiff_t(num_frequencies) * num_polarizations * num_dishes),
        first_fpga_seq_num(config.get_default<std::int64_t>(unique_name, "first_fpga_seq_num", 0)),
        noise_rms(config.get_default<float>(unique_name, "noise_rms", 0)),
        seed(config.get_default<std::uint64_t>(unique_name, "seed", 1)),
        noise_period(config.get_default<std::int64_t>(unique_name, "noise_period_frames", 4)
                     * num_times) {
        if (num_polarizations > 2)
            FATAL_ERROR_NON_OO("testSkySignal supports at most 2 polarizations");

        for (const auto& s :
             config.get_default<nlohmann::json>(unique_name, "sources", nlohmann::json::array())) {
            Source src;
            src.amplitude = s.at("amplitude").get<float>();
            src.n_x = s.at("direction").at(0).get<double>();
            src.n_y = s.at("direction").at(1).get<double>();
            src.tone_cycles_per_frame = s.at("tone_cycles_per_frame").get<int>();
            src.pol[0] = s.contains("pol") ? s.at("pol").at(0).get<float>() : 1.0f;
            src.pol[1] = s.contains("pol") ? s.at("pol").at(1).get<float>() : 1.0f;
            src.windowed = s.contains("window");
            src.window_begin = src.windowed ? s.at("window").at(0).get<std::int64_t>() : 0;
            src.window_end = src.windowed ? s.at("window").at(1).get<std::int64_t>() : 0;
            if (src.n_x * src.n_x + src.n_y * src.n_y > 1)
                FATAL_ERROR_NON_OO("testSkySignal: source direction is not on the unit sphere");
            sources.push_back(src);
        }

        // Which inputs carry signal
        std::vector<bool> active(std::size_t(num_polarizations) * num_dishes, true);
        if (config.exists(unique_name, "active_inputs")) {
            std::fill(active.begin(), active.end(), false);
            for (const auto& dp :
                 config.get<std::vector<std::vector<int>>>(unique_name, "active_inputs")) {
                if (dp.size() != 2 || dp[0] < 0 || dp[0] >= num_dishes || dp[1] < 0
                    || dp[1] >= num_polarizations)
                    FATAL_ERROR_NON_OO("testSkySignal: bad entry in `active_inputs`");
                active.at(dp[1] * num_dishes + dp[0]) = true;
            }
        }

        // Dish positions in the grid frame
        const double dx = config.get<double>("/telescope", "dish_separation_x_m");
        const double dy = config.get<double>("/telescope", "dish_separation_y_m");
        std::vector<bool> exists(num_dishes, false);
        std::vector<double> pos_x(num_dishes), pos_y(num_dishes), pos_z(num_dishes);
        for (const auto& dish : config.get<nlohmann::json>("/telescope", "dish_inputs")) {
            const int d = dish.at("dish_idx").get<int>();
            if (d < 0 || d >= num_dishes)
                FATAL_ERROR_NON_OO("testSkySignal: dish_idx {:d} out of range", d);
            const auto& disp = dish.at("feed_pos_disp_m");
            exists.at(d) = true;
            pos_x.at(d) = dx * dish.at("grid_x_idx").get<int>() + disp.at(0).get<double>();
            pos_y.at(d) = dy * dish.at("grid_y_idx").get<int>() + disp.at(1).get<double>();
            pos_z.at(d) = disp.at(2).get<double>();
        }

        // Per-input noise switch
        has_noise.resize(num_elements);
        for (std::ptrdiff_t f = 0; f < num_frequencies; ++f)
            for (int p = 0; p < num_polarizations; ++p)
                for (int d = 0; d < num_dishes; ++d)
                    has_noise[(f * num_polarizations + p) * num_dishes + d] =
                        noise_rms > 0 && exists[d] && active[p * num_dishes + d];

        // Geometric phases, amplitudes and polarization weights per source and input
        const double c0 = 299792458.0; // [m/s]
        const double channel_width = config.get<double>(unique_name, "adc_frequency")
                                     / config.get<double>(unique_name, "num_samples_per_frame");
        geometry.resize(sources.size() * num_elements);
        for (std::size_t s = 0; s < sources.size(); ++s) {
            const Source& src = sources[s];
            const double n_z = std::sqrt(1 - src.n_x * src.n_x - src.n_y * src.n_y);
            for (std::ptrdiff_t f = 0; f < num_frequencies; ++f) {
                const double nu =
                    (frequency_channels[f] + double(src.tone_cycles_per_frame) / num_times)
                    * channel_width;
                for (int p = 0; p < num_polarizations; ++p) {
                    for (int d = 0; d < num_dishes; ++d) {
                        const std::ptrdiff_t e = (f * num_polarizations + p) * num_dishes + d;
                        const bool on = exists[d] && active[p * num_dishes + d];
                        const double delay =
                            (src.n_x * pos_x[d] + src.n_y * pos_y[d] + n_z * pos_z[d]) / c0;
                        geometry[s * num_elements + e] =
                            on ? std::complex<float>(std::polar(double(src.amplitude * src.pol[p]),
                                                                -2 * M_PI * nu * delay))
                               : 0.0f;
                    }
                }
            }
        }
    }

    const std::vector<Source>& get_sources() const {
        return sources;
    }

    /// Fill rows [t_begin, t_end) of a [T, F, P, D] frame whose first sample is `frame_seq`.
    /// With `periodic_only`, sources with a time window are left out.
    void fill(kotekan::int4x2_swapped_withoffset_t* const frame, const std::int64_t frame_seq,
              const int t_begin, const int t_end, const bool periodic_only) const {
#pragma omp parallel for
        for (int t = t_begin; t < t_end; ++t) {
            const std::int64_t seq = frame_seq + t;
            const std::int64_t rel = seq - first_fpga_seq_num;
            std::vector<std::complex<float>> phasors(sources.size());
            std::vector<std::size_t> on;
            for (std::size_t s = 0; s < sources.size(); ++s) {
                const Source& src = sources[s];
                if (src.windowed
                    && (periodic_only || seq < src.window_begin || seq >= src.window_end))
                    continue;
                const std::int64_t cycle = (rel % num_times + num_times) % num_times;
                phasors[s] = std::polar(1.0, 2 * M_PI * src.tone_cycles_per_frame * double(cycle)
                                                 / num_times);
                on.push_back(s);
            }
            const std::uint64_t noise_index =
                std::uint64_t((rel % noise_period + noise_period) % noise_period) * num_elements;
            kotekan::int4x2_swapped_withoffset_t* const row =
                frame + std::ptrdiff_t(t) * num_elements;
            for (std::ptrdiff_t e = 0; e < num_elements; ++e) {
                std::complex<float> value = 0;
                for (const std::size_t s : on)
                    value += geometry[s * num_elements + e] * phasors[s];
                if (has_noise[e])
                    value += noise(noise_index + e);
                // The arguments are (imaginary, real)
                row[e] = kotekan::int4x2_swapped_withoffset_t(quantize(value.imag()),
                                                              quantize(value.real()));
            }
        }
    }

private:
    static std::uint64_t splitmix64(std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    // Gaussian noise with standard deviation `noise_rms` per component (Box-Muller). The
    // spectral kurtosis RFI flagging needs Gaussian statistics; approximations such as a sum of a
    // few uniform variates have visibly lower kurtosis and are flagged.
    std::complex<float> noise(const std::uint64_t index) const {
        const std::uint64_t h = splitmix64(seed * 0x2545f4914f6cdd1dULL + index);
        const double u1 = (double(h >> 32) + 1) / 4294967296.0;   // (0, 1]
        const double u2 = double(h & 0xffffffffU) / 4294967296.0; // [0, 1)
        return std::polar(float(noise_rms * std::sqrt(-2 * std::log(u1))), float(2 * M_PI * u2));
    }

    static std::int8_t quantize(const float x) {
        return std::int8_t(std::clamp(long(std::lrintf(x)), -7L, +7L));
    }

    const int num_times;
    const std::ptrdiff_t num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    const std::ptrdiff_t num_elements;
    const std::int64_t first_fpga_seq_num;
    const float noise_rms;
    const std::uint64_t seed;
    const std::int64_t noise_period;

    std::vector<Source> sources;
    std::vector<std::complex<float>> geometry; // [source][F][P][D]
    std::vector<bool> has_noise;               // [F][P][D]
};

#endif // TEST_SKY_SIGNAL_HPP
