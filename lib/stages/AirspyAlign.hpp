/**
 * @file
 * @brief Stage that measures the time-lag between two airspy streams via FFT cross-correlation.
 *  - AirspyAlign : public kotekan::Stage
 */

#ifndef AIRSPY_ALIGN_HPP
#define AIRSPY_ALIGN_HPP

#include <stdint.h>             // for uint32_t
#include <condition_variable>   // for condition_variable
#include <mutex>                // for mutex
#include <string>               // for string

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "restServer.hpp"       // for connectionInstance

/**
 * @class AirspyAlign
 * @brief Kotekan stage to measure the time-lag between two real input streams.
 *
 * Consumes raw @c short samples from two airspy producers. On REST request the
 * stage captures @c lag_window samples from each input, then in the callback
 * performs zero-padded FFT cross-correlation to estimate the integer sample
 * lag between A and B. The lag, and optionally the full correlation magnitudes,
 * are returned as JSON.
 *
 * @par Buffers
 * @buffer in_bufA Input kotekan buffer A.
 *     @buffer_format Array of @c short
 *     @buffer_metadata none
 * @buffer in_bufB Input kotekan buffer B.
 *     @buffer_format Array of @c short
 *     @buffer_metadata none
 *
 * @par REST endpoints
 *  - GET @c <unique_name>/cal_lag         — returns @c {"lag": int}
 *  - GET @c <unique_name>/get_correlation — returns @c {"lag": int, "corr_pos": [...], "corr_neg":
 * [...]}
 *
 * @conf   lag_window   UInt (default: input frame size in samples). Number of samples to capture.
 *
 * @author Keith Vanderlinde
 */
class AirspyAlign : public kotekan::Stage {
public:
    AirspyAlign(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~AirspyAlign() override;

    void main_thread() override;

    void cal_lag_callback(kotekan::connectionInstance& conn);
    void get_correlation_callback(kotekan::connectionInstance& conn);

private:
    /// Shared implementation for the two callbacks.
    void start_callback(kotekan::connectionInstance& conn, bool send_corr);

    /// Input buffers.
    Buffer* buf_inA;
    Buffer* buf_inB;

    /// Local capture buffers, length @c lag_window each.
    short* localA;
    short* localB;

    /// Guards @c going and the contents of @c localA / @c localB during a capture.
    std::mutex going_mutex;
    /// Signalled by main_thread when it transitions @c going from true back to false.
    std::condition_variable going_cv;
    /// Flag flipped by REST callback to request a capture; cleared by main_thread once filled.
    bool going = false;
    /// Capture length in samples.
    uint32_t _lag_window;

    /// Frame indices.
    int frame_inA;
    int frame_inB;
};

#endif
