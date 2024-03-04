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
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for frameID, freq_ctype, input_ctype, prod_ctype

#include <cstdint>    // for uint32_t, int32_t
#include <mutex>      // for mutex
#include <string>     // for string
#include <time.h>     // for size_t, timespec
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

    /**
     * @brief The main thread function for N2kAccumulate.
     * 
     * This function is responsible for the main logic of the N2kAccumulate class.
     */
    void main_thread() override;

    /**
     * @brief Copy accumulated visibility matrix and weights to the output buffer,
     * reset the visibility and weights matrices.
     * 
     * Helper function to keep code a bit more readable.
     *
     * @param in_frame_id The input frame ID.
     * @param out_frame_id The output frame ID.
     * @return bool True if successful, false otherwise.
     */
    bool output_and_reset(int &in_frame_id, int &out_frame_id);

private:
    // Buffers to read/write
    Buffer* in_buf; /// Buffer containing input frames
    Buffer* out_buf; /// Output for the main vis dataset only

    // Parameters saved from the config files
    size_t _num_freq_in_frame;
    size_t _n_fpga_samples_per_N2k_frame;
    size_t _n_fpga_samples_N2k_integrates_for;
    size_t _n_fpga_samples_per_vis_sample;
    size_t _n_vis_samples_per_N2k_output_frame;
    size_t _n_vis_samples_per_in_frame;
    
    // Frame and vis sample durations in nanoseconds
    uint64_t _in_frame_duration_nsec;
    uint64_t _in_frame_vis_duration_nsec;
    
    size_t _num_elements;
    size_t _num_vis_products;
    size_t _num_in_frame_products;

    // The below vectors are initialized in the constructor after _num_vis_products
    // and _num_freq_in_frame are known.
    std::vector<int32_t> _vis;
    std::vector<int32_t> _vis_even;
    std::vector<int32_t> _weights;
    // number of fpga samples, per frequency, in frame
    std::vector<int32_t> _n_valid_fpga_samples_in_vis;
    std::vector<int32_t> _n_valid_fpga_samples_in_vis_even;
    std::vector<int32_t> _n_valid_sample_diff_sq_sum;

    // Reference to the prometheus metric that we will use for counting skipped
    // frames
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& skipped_frame_counter;
};

#endif
