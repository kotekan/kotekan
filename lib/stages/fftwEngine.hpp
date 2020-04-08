/**
 * @file
 * @brief An FFTW-based F-engine stage.
 *  - fftwEngine : public kotekan::Stage
 */

#ifndef FFTW_ENGINE_HPP
#define FFTW_ENGINE_HPP
#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <fftw3.h>
#include <string>
#include <unistd.h>

/**
 * @class fftwEngine
 * @brief Kotekan Stage to Fourier Transform an input stream.
 *
 * This is a simple signal processing stage which takes (complex) data from an input buffer,
 * Fourier Transforms it with FFTW, and stuffs the results into an output buffer.
 * Both input and output buffers' frame lengths should be integer multiples of the FFT length,
 * though they need not be the same length as each other.
 * Assumes I/Q (complex) data at the input.
 *
 * This producer depends on libfftw3.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer, to be consumed from.
 *     @buffer_format Array of @c shorts
 *     @buffer_metadata none
 * @buffer out_buf Output kotekan buffer, to be produced into.
 *     @buffer_format Array of @c fftwf_complex
 *     @buffer_metadata none
 *
 * @conf   spectrum_length Int. Number of samples in the input spectrum. Defaults to 1024.
 *
 * @todo    Add checking to make sure the input and output buffers' frames are
 *          appropriately sized, i.e. integer multiples of spectrum_length.
 * @todo    Add a flag to allow real inputs.
 * @todo    Add some metadata to allow different data types for in/out.
 *
 * @author Keith Vanderlinde
 *
 */
class fftwEngine : public kotekan::Stage {
public:
    /// Constructor, also initializes FFTW and values from config yaml.
    fftwEngine(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    /// Destructor, frees local allocs and exits FFTW.
    virtual ~fftwEngine();
    /// Primary loop, which waits on input frames, FFTs, and dumps to output.
    void main_thread() override;

private:
    /// Kotekan buffer which this stage consumes from.
    /// Data should be packed as int16_t values, [r,i] in each 32b value.
    struct Buffer* in_buf;
    /// Kotekan buffer which this stage produces into.
    struct Buffer* out_buf;

    /// Frame index for the input buffer.
    int frame_in;
    /// Frame index for the output buffer.
    int frame_out;

    // options
    /// FFTW buffer for staging the input samples.
    fftwf_complex* samples;
    /// FFTW buffer which returns the FFT'd results.
    fftwf_complex* spectrum;
    /// FFTW object containing information about the transform
    fftwf_plan fft_plan;
    /// Length of the FFT to run
    int spectrum_length;
};


#endif
