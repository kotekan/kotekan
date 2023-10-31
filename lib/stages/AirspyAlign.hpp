/**
 * @file
 * @brief A simple cross-correlator (sum-sq) stage.
 *  - AirspyAlign : public kotekan::Stage
 */

#ifndef SIMPLE_CROSSCORR_HPP
#define SIMPLE_CROSSCORR_HPP
#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"
#include "restServer.hpp"      // for connectionInstance

#include <string> // for string

/**
 * @class AirspyAlign
 * @brief Kotekan stage to autocorrelate a single stream of values.
 *
 * This is a simple signal processing stage which takes complex @c float2 data from an input buffer,
 * calculates the modulus squared in each spectral bin, integrates over time,
 * then stuffs the results into an output buffer.
 * Both input and output buffers' frame lengths should be integer multiples of the spectrum length,
 * though they need not be the same length as each other.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer, to be consumed from.
 *     @buffer_format Array of <tt> complex float2 </tt>
 *     @buffer_metadata none
 * @buffer out_buf Output kotekan buffer, to be produced into.
 *     @buffer_format Array of @c uint
 *     @buffer_metadata none
 *
 * @conf   spectrum_length      Int (default 1024). Number of samples in the spectrum.
 * @conf   integration_length   Int (default 1024). Number of time samples to sum.
 *
 * @todo    Convert input buffer to VDIF format?
 * @todo    Add some metadata to allow different data types for in/out.
 *
 * @author Keith Vanderlinde
 *
 */
class AirspyAlign : public kotekan::Stage {
public:
    /// Constructor, also initializes FFTW and values from config yaml.
    AirspyAlign(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Destructor, frees local allocs and exits FFTW.
    virtual ~AirspyAlign();

    /// Primary loop, which waits on input frames, FFTs, and dumps to output.
    void main_thread() override;
    void start_callback(kotekan::connectionInstance& conn);

private:
    /// Kotekan buffer which this stage consumes from.
    /// Data should be packed as complex @c float pairs.
    Buffer* buf_inA;
    Buffer* buf_inB;

    short *localA;
    short *localB;
    bool going = false;

    /// Frame index for the input buffer.
    int frame_inA;
    int frame_inB;

};


#endif
