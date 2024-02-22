/*****************************************
@file
@brief Accumulation and gating of visibility data.
- N2kAccumulate : public kotekan::Stage
*****************************************/
#ifndef N2K_ACCUMULATE_HPP
#define N2K_ACCUMULATE_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t
#include "gateSpec.hpp"          // for gateSpec
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for frameID, freq_ctype, input_ctype, prod_ctype

#include <cstdint>    // for uint32_t, int32_t
#include <deque>      // for deque
#include <functional> // for function
#include <map>        // for map
#include <memory>     // for unique_ptr
#include <mutex>      // for mutex
#include <string>     // for string
#include <time.h>     // for size_t, timespec
#include <utility>    // for pair
#include <vector>     // for vector


/**
 * @class N2kAccumulate
 * @brief Accumulate the high rate GPU output into integrated VisBuffers.
 *
 *
 * @par Buffers
 * @buffer in_buf
 *         @buffer_format GPU packed information
 *         @buffer_metadata chordMetadata
 * @buffer out_buf The accumulated and tagged data.
 *         @buffer_format VisBuffer structured.
 *         @buffer_metadata VisMetadata
 */
class N2kAccumulate : public kotekan::Stage {
public:
    N2kAccumulate(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~N2kAccumulate() = default;
    void main_thread() override;

private:
    // Buffers to read/write
    Buffer* in_buf; // Buffer containing input frames
    Buffer* out_buf; // Output for the main vis dataset only

    // Parameters saved from the config files
    size_t _num_freq_in_frame;
    size_t _n_fpga_samples_per_N2k_frame;
    size_t _n_fpga_samples_N2k_integrates_for;
    size_t _n_vis_samples_per_N2k_output_frame;
    size_t _n_vis_samples_per_in_frame;
    
    // Frame and vis sample durations in nanoseconds
    uint64_t _in_frame_duration_nsec;
    uint64_t _in_frame_vis_duration_nsec;
    
    size_t _num_vis_products;

    // Reference to the prometheus metric that we will use for counting skipped
    // frames
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& skipped_frame_counter;
};

#endif
