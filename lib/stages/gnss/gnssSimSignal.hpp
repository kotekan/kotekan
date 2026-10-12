/**
 * @file
 * @brief Simulated wideband GNSS baseband signal, a test source for the channelized despread.
 *
 * Generates the complex baseband voltage a GNSS signal presents at the F-engine input: a +-1
 * spreading code oversampled to the sample rate, at a Doppler offset, scaled by a complex
 * amplitude, plus optional complex Gaussian noise. Passed through a PFB (as the reference
 * analysis in test_gnss_channelized_despread.cpp does) it gives a channelized source for
 * testing @ref gnssChannelizedDespread.hpp without an F-engine. Has no FFTW dependency.
 */

#ifndef GNSS_SIM_SIGNAL_HPP
#define GNSS_SIM_SIGNAL_HPP

#include <complex> // for complex
#include <cstdint> // for int8_t, uint32_t
#include <vector>  // for vector

namespace gnss {

/**
 * Wideband complex baseband BPSK signal. The length-(code.size()*samp_per_chip)
 * output is @c amplitude * chip(n) * exp(i 2*pi * fd_cyc_per_sample * n) plus
 * complex Gaussian noise of per-component std @c noise_sigma/sqrt(2) (so total
 * noise power is @c noise_sigma^2). @c code holds the +-1 chips.
 */
std::vector<std::complex<float>> sim_bpsk_baseband(const std::vector<int8_t>& code,
                                                   int samp_per_chip, double fd_cyc_per_sample,
                                                   std::complex<float> amplitude,
                                                   float noise_sigma = 0.0f, uint32_t seed = 0);

} // namespace gnss

#endif // GNSS_SIM_SIGNAL_HPP
