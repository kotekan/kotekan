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

#include <fftw3.h> // for fftwf_complex, fftwf_plan, fftwf_plan_s
#include <string>  // for string

/**
 * @class fftwEngine
 * @brief Kotekan Stage to Fourier Transform an input stream of samples.
 *
 * Reads samples from an input buffer, FFTs them with FFTW, and writes the
 * resulting complex spectra into an output buffer. Both input and output
 * buffers' frame lengths must be integer multiples of one FFT's worth of data.
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
 * @conf   spectrum_length  Int (default 128). Number of complex bins per output spectrum.
 * @conf   input_type       String (default "complex"). One of "complex" or "real".
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
};

#endif
