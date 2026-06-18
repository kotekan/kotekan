/**
 * @file
 * @brief Simulated wideband GNSS baseband signal -- the source side of the
 *        channelized-despread development path.
 *
 * Generates the complex baseband voltage a GNSS signal would present at the F
 * engine input: a +-1 spreading code, oversampled to the sample rate, carried
 * at a Doppler offset, scaled by a complex amplitude, plus optional complex
 * Gaussian noise. Feeding this through the PFB channelizer (@ref fftwEngine with
 * num_taps>1, or the reference analysis in the unit test) yields a simulated
 * channelized source for exercising @ref gnssChannelizedDespread without the
 * real X-engine F-engine. Pure and FFTW-free so it is reusable by tests and a
 * future sim-source stage alike.
 */

#ifndef GNSS_SIM_SIGNAL_HPP
#define GNSS_SIM_SIGNAL_HPP

#include <complex>  // for complex
#include <cstdint>  // for int8_t, uint32_t
#include <vector>   // for vector

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
