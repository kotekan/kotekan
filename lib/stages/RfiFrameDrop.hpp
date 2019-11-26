/*****************************************
@file
@brief Drops GPU frames when they are contaminated with RFI
- RfiFrameDrop : public kotekan::Stage
*****************************************/
#ifndef VALVE_HPP
#define VALVE_HPP

#include "Config.hpp"
#include "prometheusMetrics.hpp"
#include "Stage.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"

#include "json.hpp"

#include <string>


/**
 * @brief Drop whole GPU frames based on the high cadence kurtosis statistics.
 *
 * This stage reads the kurtosis data output and uses it to filter the GPU N^2
 * data that feeds through the system. This is complex because the subframe
 * processing of the N^2 data, the kurtosis frames contain the data
 * corresponding to multiple N^2 frames and so we need to be careful to
 * synchronize them.
 *
 * @par Buffers
 * @buffer  in_buf_vis        The kotekan buffer from which frames are read, can be any size.
 *          @buffer_format    GPU packed upper triangle
 *          @buffer_metadata  chimeMetadata
 * @buffer  in_buf_sk         The high cadence Spectral Kurtosis estimates.
 *          @buffer_format    High cadence SK values.
 *          @buffer_metadata  chimeMetadata
 * @buffer  out_buf           The filtered GPU visibility data.
 *          @buffer_format    GPU packed upper triangle
 *          @buffer_metadata  chimeMetadata
 *
 * @conf  num_elements          The number of inputs to the correlator.
 * @conf  num_sub_frames        The number of N^2 frames we will receive per SK frame.
 * @conf  samples_per_data_set  The number of FPGA samples combined into an SK frame.
 * @conf  sk_step               The number of FPGA samples combined into an SK element.
 * @conf  thresholds            List of dicts containing `fraction` and `threshold` keys.
 *                              Each of these is a float giving the *fraction* of SK
 *                              values that must exceed the *threshold* for the frame
 *                              to get dropped.
 *
 * @par Metrics
 * @metric  kotekan_rfiframedrop_dropped_frame_total
 *          The number of frames dropped by each criteria.
 * @metric  kotekan_rfiframedrop_frame_total
 *          The total number of frames seen by this stage per frequency.
 *
 * @author  Richard Shaw
 *
 */
class RfiFrameDrop : public kotekan::Stage {

public:
    /// Constructor.
    RfiFrameDrop(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Primary loop.
    void main_thread() override;

private:
    /// Copy a frame from the input buffer to the output buffer.
    static void copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest, int frame_id_dest);

    /// Parse thresholds (returns false if the parsing failed)
    bool parse_thresholds(nlohmann::json j);

    /// Input buffer
    Buffer* _buf_in_vis;
    Buffer* _buf_in_sk;

    /// Output buffer to receive baseline subset visibilities
    Buffer* _buf_out;

    /// Thresholds
    std::vector<std::tuple<float, size_t, float>> _thresholds;

    /// Prometheus metrics to export
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& _dropped_frame_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& _frame_counter;

    /// Toggle RFI zeroing
    bool enable_rfi_zero;

    size_t num_elements;
    size_t num_sub_frames;
    size_t sk_step;
    size_t sk_samples_per_frame;
};


#endif // VALVE_HPP
