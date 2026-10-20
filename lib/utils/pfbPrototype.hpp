/**
 * @file
 * @brief Polyphase filter bank (PFB) prototype filter and polyphase fold.
 *
 * A PFB runs each block of @c num_chan samples through a polyphase decomposition of a
 * length-(num_chan * num_taps) lowpass prototype before the num_chan-point DFT. Compared with a
 * plain DFT of the block, whose channels have -13 dB sidelobes, the channels get steep edges and a
 * deep stopband set by the prototype's window.
 *
 * The functions here are the FFT-free part: the prototype design and the fold
 * @f$ out[p] = \sum_q h[qN+p]\,x[t_0+qN+p] @f$ over the last N*P samples (from @f$ t_0 @f$, in
 * time order), whose N-point forward DFT gives the channels in the same order as a plain DFT of
 * the newest block. Used by @ref fftwEngine.
 */

#ifndef PFB_PROTOTYPE_HPP
#define PFB_PROTOTYPE_HPP

#include <cstring> // for memmove, memcpy
#include <string>  // for string
#include <vector>  // for vector

/// Window applied to the prototype's sinc.
enum class PfbWindow { Rectangular, Hann, Hamming, Blackman };

/**
 * @brief Design a length-(num_chan * num_taps) lowpass prototype: a sinc with its cutoff at the
 *        channel spacing, times @p window.
 *
 * The taps are scaled to sum to @p num_chan, so a tone at a channel centre has the gain it has in
 * a plain DFT of the block. Throws std::invalid_argument if the prototype would have fewer than
 * two taps.
 *
 * @param num_chan  Number of channels (the DFT length).
 * @param num_taps  Taps per channel.
 * @param window    Window applied to the sinc.
 */
std::vector<float> pfb_prototype(int num_chan, int num_taps, PfbWindow window = PfbWindow::Hamming);

/// Map "rect", "hann", "hamming" or "blackman" to a @ref PfbWindow; throws
/// std::invalid_argument for any other name.
PfbWindow pfb_window_from_string(const std::string& name);

/**
 * @brief Shift the next @p num_chan input samples into the delay line.
 *
 * @param hist      Delay line of length num_chan * num_taps, in time order.
 * @param block     The next @p num_chan samples, in time order.
 * @param num_chan  Number of channels.
 * @param num_taps  Taps per channel.
 */
template<typename T>
inline void pfb_push(T* hist, const T* block, int num_chan, int num_taps) {
    std::memmove(hist, hist + num_chan, (num_chan * (num_taps - 1)) * sizeof(T));
    std::memcpy(hist + num_chan * (num_taps - 1), block, num_chan * sizeof(T));
}

/**
 * @brief Fold the delay line with the prototype: out[p] = sum_q proto[qN+p] * hist[qN+p].
 *
 * The forward (sign -1) num_chan-point DFT of @p out gives the channels.
 *
 * @param hist      Delay line from @ref pfb_push, in time order.
 * @param proto     Prototype from @ref pfb_prototype.
 * @param num_chan  Number of channels.
 * @param num_taps  Taps per channel.
 * @param out       Output, length @p num_chan.
 */
template<typename T>
inline void pfb_fold(const T* hist, const float* proto, int num_chan, int num_taps, T* out) {
    for (int p = 0; p < num_chan; ++p) {
        T acc{};
        for (int q = 0; q < num_taps; ++q)
            acc += proto[q * num_chan + p] * hist[q * num_chan + p];
        out[p] = acc;
    }
}

#endif // PFB_PROTOTYPE_HPP
