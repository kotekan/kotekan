/**
 * @file
 * @brief Polyphase filter bank (PFB) core: prototype filter design + the
 *        polyphase fold that precedes the channelizing FFT.
 *
 * A straight-FFT channelizer (@ref fftwEngine) is a length-N DFT per block: its
 * per-channel response is a Dirichlet/sinc kernel with ~-13 dB first sidelobes,
 * ~3.9 dB scalloping loss, and heavy inter-channel leakage. A PFB instead runs
 * the block through a polyphase decomposition of a length-(N*P) lowpass
 * prototype before the same N-point DFT, giving near-brickwall channels: a flat
 * passband and a deep stopband set by the prototype window. That isolation is
 * what lets us recombine the handful of channels covering a GNSS carrier (see
 * @ref gnssBandPlan.hpp) and despread per-channel without inter-channel
 * contamination.
 *
 * This header is the pure, FFTW-free DSP: prototype design and the fold
 * @f$ v[p] = \sum_q h[qN+p]\,x[mN-p-qN] @f$ whose N-point DFT yields the
 * channels. The FFTW wrapper that turns @c v into channels lives in the
 * channelizer stage. Kept separate so the tricky math is unit-tested against a
 * reference DFT with no FFTW dependency.
 */

#ifndef PFB_PROTOTYPE_HPP
#define PFB_PROTOTYPE_HPP

#include <complex>  // for complex
#include <cstring>  // for memmove
#include <string>   // for string
#include <vector>   // for vector

namespace dsp {

/// Window applied to the prototype's windowed-sinc. Higher-suppression windows
/// trade main-lobe width (channel transition sharpness) for stopband depth.
enum class Window {
    Rectangular, ///< none: -13 dB sidelobes (mainly for reference/tests)
    Hann,        ///< -31 dB
    Hamming,     ///< -42 dB (default)
    Blackman,    ///< -58 dB
    Kaiser,      ///< tunable via beta
};

/**
 * Design a length-(num_chan*num_taps) lowpass prototype: a sinc with cutoff at
 * the channel spacing (pi/num_chan) times the chosen window, normalized so the
 * taps sum to num_chan -- i.e. each polyphase branch has unit gain, so the
 * channelizer's passband gain (and output power) matches a straight FFT
 * regardless of num_taps. `num_taps` (P) is the polyphase depth -- taps per
 * channel; P=1 degenerates toward a plain FFT, P>=4 gives good isolation.
 * `kaiser_beta` is used only for Window::Kaiser.
 */
std::vector<float> pfb_prototype(int num_chan, int num_taps, Window window = Window::Hamming,
                                 double kaiser_beta = 8.0);

/// Map a config window name ("hamming", "hann", "blackman", "kaiser", "rect")
/// to a @ref Window; throws std::invalid_argument on an unknown name.
Window window_from_string(const std::string& name);

/**
 * Advance the polyphase delay line by one hop. `hist` is the length
 * num_chan*num_taps line, newest first (hist[0] = newest); `block` is the next
 * num_chan input samples in chronological order (block[num_chan-1] is newest).
 * Older samples slide back num_chan slots; the block is inserted newest-first.
 * Templated for real (float) or complex samples.
 */
template<typename T>
inline void pfb_push(T* hist, const T* block, int num_chan, int num_taps) {
    const int L = num_chan * num_taps;
    std::memmove(hist + num_chan, hist, (L - num_chan) * sizeof(T));
    for (int i = 0; i < num_chan; ++i)
        hist[i] = block[num_chan - 1 - i]; // newest sample -> hist[0]
}

/**
 * Polyphase fold for one hop. `hist` is the delay line of the most recent
 * num_chan*num_taps input samples, newest first (hist[0] = x[mN], hist[j] =
 * x[mN-j]); `proto` is the prototype from @ref pfb_prototype. Writes the
 * length-num_chan pre-FFT vector @c out, where
 * @f$ out[p] = \sum_{q=0}^{P-1} proto[qN+p]\,hist[qN+p] @f$. The caller takes
 * the forward (sign -1) N-point DFT of @c out to get the channels. Templated
 * for real or complex samples.
 */
template<typename T>
inline void pfb_fold(const T* hist, const float* proto, int num_chan, int num_taps, T* out) {
    for (int p = 0; p < num_chan; ++p) {
        T acc{};
        for (int q = 0; q < num_taps; ++q) {
            const int idx = q * num_chan + p;
            acc += proto[idx] * hist[idx];
        }
        out[p] = acc;
    }
}

} // namespace dsp

#endif // PFB_PROTOTYPE_HPP
