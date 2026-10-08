/**
 * @file
 * @brief An FFTW-based F-engine stage.
 *  - fftwEngine : public kotekan::Stage
 */

#ifndef FFTW_ENGINE_HPP
#define FFTW_ENGINE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <complex> // for complex
#include <fftw3.h> // for fftwf_complex, fftwf_plan, fftwf_plan_s
#include <string>  // for string
#include <vector>  // for vector

/**
 * @class fftwEngine
 * @brief Kotekan Stage to Fourier Transform an input stream of samples.
 *
 * Reads samples from an input buffer, FFTs them with FFTW, and writes the
 * resulting complex spectra into an output buffer. Each input frame must hold a
 * whole number of transforms, and an output frame is twice the input frame's size.
 *
 * Two input formats are supported, selectable via @c input_type:
 *  - @c complex (default): packed int16 I/Q pairs. Each FFT consumes
 *    @c spectrum_length complex samples and emits @c spectrum_length complex
 *    bins. The output is fftshifted -- the two halves of the raw FFT are
 *    swapped (memcpy of @c spectrum_length/2 bins each way) so DC lands at
 *    index @c spectrum_length/2 and bins run monotonically from -Fs/2 up to
 *    +Fs/2.
 *  - @c real: packed int16 real samples. Each FFT consumes
 *    @c 2*spectrum_length real samples and emits @c spectrum_length complex
 *    bins (the first half of an r2c transform; Nyquist bin discarded). No
 *    fftshift here -- bins run 0 (DC) up to just below Fs/2.
 *
 * With @c num_taps > 1 each transform is preceded by the polyphase fold of a
 * windowed-sinc prototype filter (see pfbPrototype.hpp), making the stage a
 * polyphase filter bank. With 4 taps and the hamming window, a tone leaks about
 * -50 dB into channels a channel spacing or more from it, against the plain
 * FFT's -13 dB sidelobes; blackman leaks more next to the channel and less from about 1.5
 * channels out. A tone at a channel centre keeps the plain FFT's gain, a tone
 * at a channel edge is 6 dB down in each neighbour (3.9 dB for the plain FFT),
 * and white noise per channel is about 1 dB lower. The fold reads a delay line
 * of the last @c num_taps transforms' input, carried across frames, so each
 * spectrum is centred (num_taps - 1) / 2 transforms earlier than the plain
 * FFT's, and the first num_taps - 1 spectra after startup see zeros for the
 * missing history.
 *
 * Depends on libfftw3.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer.
 *     @buffer_format Array of @c int16_t (real samples, or interleaved I/Q pairs)
 *     @buffer_metadata none
 * @buffer out_buf Output kotekan buffer.
 *     @buffer_format Array of @c fftwf_complex
 *     @buffer_metadata none
 *
 * @conf   spectrum_length  Int (default 128). Number of complex bins per output spectrum;
 *                          even for complex input.
 * @conf   input_type       String (default "complex"). One of "complex" or "real".
 * @conf   num_taps         Int (default 1). Polyphase taps per channel; 1 is the plain FFT.
 * @conf   pfb_window       String (default "hamming"). Prototype window when num_taps > 1:
 *                          "rect", "hann", "hamming" or "blackman".
 *
 * @author Keith Vanderlinde
 */
class fftwEngine : public kotekan::Stage {
public:
    fftwEngine(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~fftwEngine() override;

    void main_thread() override;

private:
    /// Input buffer. Format depends on @c input_type (complex int16 pairs or real int16).
    Buffer* in_buf;
    /// Output buffer of @c fftwf_complex bins.
    Buffer* out_buf;

    /// Frame indices.
    int frame_in;
    int frame_out;

    /// Number of complex output bins per FFT.
    int _spectrum_length;
    /// Whether the input is real or complex.
    bool _real_input;

    /// FFTW work buffers; @c real_samples is used in real mode, @c complex_samples in complex mode.
    float* real_samples;
    fftwf_complex* complex_samples;
    /// Output of the FFT (size depends on mode).
    fftwf_complex* spectrum;
    /// FFTW plan for the configured mode.
    fftwf_plan fft_plan;

    /// Polyphase taps per channel; 1 is the plain FFT.
    int _num_taps;
    /// Prototype filter, length fft_len * @c _num_taps; empty when @c _num_taps is 1.
    std::vector<float> _proto;
    /// Polyphase delay line in time order, and one transform's input block. Only the pair for
    /// the configured input type is allocated.
    std::vector<float> _hist_real, _block_real;
    std::vector<std::complex<float>> _hist_complex, _block_complex;
};

#endif
