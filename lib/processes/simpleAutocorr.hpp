/**
 * @file simpleAutocorr.hpp
 * @brief A simple autocorrelator (sum-sq) process.
 *  - simpleAutocorr : public KotekanProcess
 **/

#ifndef SIMPLE_AUTOCORR_HPP
#define SIMPLE_AUTOCORR_HPP
#include <unistd.h>

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <string>
using std::string;

/**
 * @class simpleAutocorr
 * @brief Kotekan Process to autocorrelate a single stream of values.
 *
 * This is a simple signal processing block which takes complex @c float2 data from an input buffer,
 * calculates the modulus squared in each spectral bin, integrates over time,
 * then stuffs the results into an output buffer.
 * Both input and output buffers' frame lengths should be integer multiples of the spectrum length,
 * though they need not be the same length as each other.
 *
 * This producer depends on libairspy.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer, to be consumed from.
 *     @buffer_format Array of <tt> complex float2 </tt>
 *     @buffer_metadata none
 * @buffer out_buf Output kotekan buffer, to be produced into.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   in_buf               Buffer.  Input kotekan buffer, to be consumed from.
 * @conf   out_buf              Buffer. Output kotekan buffer, to be produced into.
 * @conf   spectrum_length      Int. Number of samples in the spectrum. Defaults to 1024.
 * @conf   integration_length   Int. Number of time samples to sum. Defaults to 1024.
 *
 * @todo    Convert input buffer to VDIF format?
 * @todo    Add some metadata to allow different data types for in/out.
 *
 * @author Keith Vanderlinde
 *
 **/
class simpleAutocorr : public KotekanProcess {
public:
    /// Constructor, also initializes FFTW and values from config yaml.
    simpleAutocorr(Config& config, const string& unique_name,
                         bufferContainer &buffer_container);

    /// Destructor, frees local allocs and exits FFTW.
    virtual ~simpleAutocorr();

    /// Primary loop, which waits on input frames, FFTs, and dumps to output.
    void main_thread();

    /// Re-parse config, not yet implemented.
    virtual void apply_config(uint64_t fpga_seq);

private:
    /// Kotekan buffer which this process consumes from.
    /// Data should be packed as comples @c float pairs.
    struct Buffer *buf_in;
    /// Kotekan buffer which this process produces into.
    struct Buffer *buf_out;

    /// Frame index for the input buffer.
    int frame_in;
    /// Frame index for the output buffer.
    int frame_out;

    //options
    /// Length of the spectrum being autocorrelated.
    int spectrum_length;
    /// Number of samples to integrate per output.
    int integration_length;
    /// Buffer for accumulating and staging the output.
    float *spectrum_out;
};


#endif
