/**
 * @file
 * @brief A simple autocorrelator (sum-sq) stage.
 *  - SampleAutocorr : public kotekan::Stage
 */

#ifndef SAMPLE_AUTOCORR_HPP
#define SAMPLE_AUTOCORR_HPP
#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "restServer.hpp"
#include "bufferContainer.hpp"

#include <string> // for string

/**
 * @class SampleAutocorr
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
class SampleAutocorr : public kotekan::Stage {
public:
    /// Constructor, also initializes FFTW and values from config yaml.
    SampleAutocorr(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Destructor, frees local allocs and exits FFTW.
    virtual ~SampleAutocorr();

    /// Primary loop, which waits on input frames, FFTs, and dumps to output.
    void main_thread() override;

    void rest_callback(kotekan::connectionInstance& conn,
                       nlohmann::json& json_request);

private:
    /// Kotekan buffer which this stage consumes from.
    /// Data should be packed as complex @c float pairs.
    struct Buffer* buf_in;
    /// Kotekan buffer which this stage produces into.
    struct Buffer* buf_out;

    /// Frame index for the input buffer.
    int frame_in;
    /// Frame index for the output buffer.
    int frame_out;

    // options
    /// Length of the spectrum being autocorrelated.
    int spectrum_length;
    /// Number of samples to integrate per output.
    int integration_length;
    /// Buffer for accumulating and staging the output.
    float* spectrum_out;

    std::string endpoint;

    bool sample_active = false;
};


#endif
