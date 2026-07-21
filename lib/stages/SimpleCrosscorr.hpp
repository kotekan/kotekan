/**
 * @file
 * @brief A simple two-input cross-correlator stage.
 *  - SimpleCrosscorr : public kotekan::Stage
 */

#ifndef SIMPLE_CROSSCORR_HPP
#define SIMPLE_CROSSCORR_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @class SimpleCrosscorr
 * @brief Kotekan stage to compute four spectral products (AA, BB, Re{AB*},
 *        Im{AB*}) from two complex-float input streams and integrate over time.
 *
 * Both input buffers' frame lengths must be integer multiples of the spectrum
 * length. Per integration the stage emits @c 4 * spectrum_length floats
 * followed by a single @c uint32_t holding the integration count, into the
 * output buffer.
 *
 * @par Buffers
 * @buffer in_bufA Input kotekan buffer A.
 *     @buffer_format Array of <tt>complex float</tt> (interleaved float pairs)
 *     @buffer_metadata none
 * @buffer in_bufB Input kotekan buffer B.
 *     @buffer_format Array of <tt>complex float</tt> (interleaved float pairs)
 *     @buffer_metadata none
 * @buffer out_buf Output kotekan buffer.
 *     @buffer_format Array of <tt>float</tt> + trailing @c uint32_t per integration
 *     @buffer_metadata none
 *
 * @conf   spectrum_length      Int (default 1024). Number of samples in the spectrum.
 * @conf   integration_length   Int (default 1024). Number of time samples to sum.
 *
 * @author Keith Vanderlinde
 */
class SimpleCrosscorr : public kotekan::Stage {
public:
    SimpleCrosscorr(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    ~SimpleCrosscorr() override;

    void main_thread() override;

private:
    /// Input buffers; complex float pairs.
    Buffer* buf_inA;
    Buffer* buf_inB;
    /// Output buffer; AA, BB, Re{AB*}, Im{AB*} per spectral bin, plus count.
    Buffer* buf_out;

    /// Frame indices.
    int frame_inA;
    int frame_inB;
    int frame_out;

    /// Length of the spectrum being correlated.
    uint32_t _spectrum_length;
    /// Number of time samples summed per output.
    uint32_t _integration_length;
    /// Accumulator buffer: spectrum_length * 4 floats.
    float* spectrum_out;
};

#endif
