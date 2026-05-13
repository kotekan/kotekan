/**
 * @file
 * @brief A simple autocorrelator (sum-sq) stage with REST one-shot sampling.
 *  - SampleAutocorr : public kotekan::Stage
 */

#ifndef SAMPLE_AUTOCORR_HPP
#define SAMPLE_AUTOCORR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include "json.hpp" // for json

#include <condition_variable> // for condition_variable
#include <mutex>              // for mutex
#include <string>             // for string

/**
 * @class SampleAutocorr
 * @brief Kotekan stage to accumulate the modulus-squared of a complex
 *        sample stream into a power spectrum, served on demand via REST.
 *
 * Consumes complex @c float pairs from a single input buffer.
 * When a client POSTs to the stage's @c /spectrum endpoint with an
 * @c integration_length value, the stage integrates that many sample-frames
 * worth of |x|^2, then returns the resulting power spectrum as JSON.
 * The stage has no output buffer.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer, to be consumed from.
 *     @buffer_format Array of <tt>complex float</tt> (interleaved float pairs)
 *     @buffer_metadata none
 *
 * @par REST endpoints
 *  - POST @c <unique_name>/spectrum with @c {"integration_length": N}
 *    returns @c {"stokes_i": [...]} once integration completes.
 *
 * @conf   spectrum_length      Int (default 1024). Number of samples in the spectrum.
 * @conf   integration_length   Int (default 1024). Default number of time samples to sum.
 *
 * @author Keith Vanderlinde
 */
class SampleAutocorr : public kotekan::Stage {
public:
    SampleAutocorr(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~SampleAutocorr() override;

    void main_thread() override;

    void rest_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

private:
    /// Input buffer; complex float pairs.
    Buffer* buf_in;

    /// Frame index for the input buffer.
    int frame_in;

    /// Length of the spectrum being autocorrelated.
    int _spectrum_length;
    /// Number of samples to integrate per output.
    int _integration_length;
    /// Buffer for accumulating and staging the output spectrum.
    float* spectrum_out;

    /// REST endpoint path.
    std::string endpoint;

    /// Guards @c sample_active / @c sample_ready / @c spectrum_out and the integration counter.
    std::mutex sample_mutex;
    /// Signalled by main_thread when @c sample_ready transitions to true.
    std::condition_variable sample_cv;
    /// Set by REST thread to request a sample; cleared by main_thread when integration completes.
    bool sample_active = false;
    /// Set by main_thread when integration is complete; cleared by REST thread on consume.
    bool sample_ready = false;
};

#endif
