/*****************************************
@file
@brief Stage for compressing visibility data.
- baselineCompression
*****************************************/
#ifndef VIS_COMPRESSION_HPP
#define VIS_COMPRESSION_HPP

#include <cstdint>                // for uint32_t
#include <functional>             // for function
#include <map>                    // for map
#include <mutex>                  // for mutex
#include <string>                 // for string
#include <thread>                 // for thread
#include <tuple>                  // for tuple
#include <utility>                // for pair
#include <vector>                 // for vector

#include "Config.hpp"             // for Config
#include "Stage.hpp"              // for Stage
#include "buffer.h"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "datasetManager.hpp"     // for dset_id_t, state_id_t, fingerprint_t
#include "datasetState.hpp"       // for prodState, stackState
#include "prometheusMetrics.hpp"  // for MetricFamily, Gauge, Counter
#include "visUtil.hpp"            // for frameID, rstack_ctype, input_ctype, prod_ctype

/**
 * @brief Compress visibility data by stacking together equivalent baselines.
 *
 * This task takes a stream of uncompressed visibility data and performs a
 * (weighted) average over the equivalent baselines.
 *
 * @note This task requires there to be an inputState and prodState registered
 *       for the incoming dataset.
 *
 * @par Buffers
 * @buffer in_bufs The input uncompressed data.
 *         @buffer_format VisBuffer structured.
 *         @buffer_metadata VisMetadata
 * @buffer out_buf The merged and transformed buffer
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf stack_type             String. Type of stacking to apply to the data.
 *                              Look at documentation of stack_X functions for
 *                              details.
 * @conf exclude_inputs         List of ints. Extra inputs to exclude from
 *                              stack.
 *
 * @par Metrics
 * @metric kotekan_baselinecompression_residuals
 *      The variance of the residuals.
 * @metric kotekan_baselinecompression_time_seconds
 *      The time elapsed to process one frame.
 * @metric kotekan_baselinecompression_frame_total
 *      Number of frames seen by each thread.
 *
 * @author Richard Shaw
 */
class baselineCompression : public kotekan::Stage {

public:
    // Default constructor
    baselineCompression(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);

    // Main loop for the stage: Creates n threads that do the compression.
    void main_thread() override;

private:
    /// Entrancepoint for n threads. Each thread takes frames with a
    /// different frame_id from the buffer and compresses them.
    void compress_thread(uint32_t thread_id);

    /// Tracks input dataset ID and gets output dataset IDs from manager
    void change_dataset_state(dset_id_t input_ds_id);

    /// Vector to hold the thread handles
    std::vector<std::thread> thread_handles;

    // The extra inputs we are excluding
    std::vector<uint32_t> exclude_inputs;

    // Alias for the type of a stack definition function. Damn C++ is verbose :)
    // Return is a pair of (num stacks total, stack_map), where stack_map is a
    // vector of (stack_output_index, conjugate) pairs. In this conjugate
    // describes whether the input must be complex conjugated prior to stacking.
    // Inputs are the usual input map and product map.
    using stack_def_fn = std::function<std::pair<uint32_t, std::vector<rstack_ctype>>(
        const std::vector<input_ctype>&, const std::vector<prod_ctype>&)>;

    /// Map from the name of a stack to its definiting function.
    std::map<std::string, stack_def_fn> stack_type_defs;

    /// The stack function to use.
    stack_def_fn calculate_stack;

    /// Number of parallel threads accessing the same buffers (default 1)
    uint32_t num_threads;

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf;

    // Frame IDs, shared by compress threads and their mutex.
    frameID frame_id_in;
    frameID frame_id_out;
    std::mutex m_frame_ids;
    std::mutex m_dset_map;

    // Map the incoming ID to an outgoing one
    std::map<dset_id_t, std::tuple<dset_id_t, const stackState*, const prodState*>> dset_id_map;

    // Map from the critical incoming states to the correct stackState
    std::map<fingerprint_t, std::tuple<state_id_t, const stackState*, const prodState*>> state_map;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& compression_residuals_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& compression_time_seconds_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& compression_frame_counter;
};


#endif
