#include "gnssSimSignal.hpp"

#include <cmath>   // for cos, sin, M_PI
#include <random>  // for mt19937, normal_distribution

namespace gnss {

std::vector<std::complex<float>> sim_bpsk_baseband(const std::vector<int8_t>& code,
                                                   int samp_per_chip, double fd_cyc_per_sample,
                                                   std::complex<float> amplitude, float noise_sigma,
                                                   uint32_t seed) {
    const size_t ns = code.size() * static_cast<size_t>(samp_per_chip);
    std::vector<std::complex<float>> sig(ns);

    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, noise_sigma * 0.70710678f); // per component

    for (size_t n = 0; n < ns; ++n) {
        const int8_t chip = code[n / samp_per_chip];
        const double ph = 2.0 * M_PI * fd_cyc_per_sample * static_cast<double>(n);
        const std::complex<float> carrier(static_cast<float>(std::cos(ph)),
                                          static_cast<float>(std::sin(ph)));
        std::complex<float> s = amplitude * static_cast<float>(chip) * carrier;
        if (noise_sigma > 0.0f)
            s += std::complex<float>(gauss(rng), gauss(rng));
        sig[n] = s;
    }
    return sig;
}

} // namespace gnss
