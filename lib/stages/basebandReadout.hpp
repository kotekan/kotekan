/*****************************************
@file
@brief Stages for triggered baseband recording
- basebandDumpData
- basebandReadout : public kotekan::Stage
*****************************************/
#ifndef BASEBAND_READOUT_H
#define BASEBAND_READOUT_H

#include "BipBuffer.hpp"              // for BipBuffer
#include "Config.hpp"                 // for Config
#include "Stage.hpp"                  // for Stage
#include "basebandReadoutManager.hpp" // for basebandDumpData, basebandReadoutManager, baseband...
#include "bufferContainer.hpp"        // for bufferContainer
#include "prometheusMetrics.hpp"      // for MetricFamily, Counter, Gauge
#include "visUtil.hpp"                // for input_ctype

#include <mutex>    // for mutex
#include <stddef.h> // for size_t
#include <stdint.h> // for int64_t, uint32_t, uint64_t
#include <string>   // for string
#include <vector>   // for vector


constexpr size_t TARGET_CHUNK_SIZE = 1024 * 1024;


/**
 * @class basebandReadout
 * @brief Stage for extracting one frequency from the baseband buffer and putting it into a new frame.
 *
 * This task manages a kotekan buffer, keeping it mostly full such that it subsets
 * of the data can be written upon triggered request.
 *
 * @par Buffers
 * @buffer in_buf buffer to manage and read. Must be several frames larger than
 *                ``num_frames_buffer`` config parameter.
 *         @buffer_format DPDK baseband ``samples_per_data_set x num_elements`` bytes
 *         @buffer_metadata chimeMetadata
 *
 * @buffer out_buf The extracted single frequency baseband output
 *         @buffer_format DPDK baseband ``samples_per_data_set x num_elements`` bytes
 *         @buffer_metadata BasebandMetadata
 *
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in the
 *                              correlator data.
 * @conf  samples_per_data_set  Int. The number of time samples in a frame.
 * @conf  max_dump_samples      Int, default 2^30. Maximum number of samples in
 *                              baseband dump. Memory used for dumps limited to
 *                              3 x num_elements x this_number.
 * @conf  num_frames_buffer     Int. Number of buffer frames to simultaneously keep
 *                              full of data. Should be few less than in_buf length.
 *
 * @par Metrics
 * @metric kotekan_baseband_readout_total
 *         The count of requests handled by an instance of this stage.
 *         Labels:
 *         - status: 'done', 'error', 'no_data'
 *         - freq_id: channel frequency received by this stage
 *
 * @author Kiyoshi Masui, Davor Cubranic
 */
class basebandReadout : public kotekan::Stage {
public:
    basebandReadout(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    virtual ~basebandReadout() = default;
    void main_thread() override;

private:
    // settings from the config file
    int _num_frames_buffer;
    int _num_elements;
    int _samples_per_data_set;
    int64_t _max_dump_samples;
    std::vector<input_ctype> _inputs;

    struct Buffer* in_buf;
    int next_frame, oldest_frame;
    std::vector<std::mutex> frame_locks;

    struct Buffer* out_buf;
    frameID out_frame_id;

    std::mutex manager_lock;

    /**
     * @brief Process incoming requests by copying the baseband data from the ring buffer
     */
    void readout_thread(const uint32_t freq_id, kotekan::basebandReadoutManager& readout_manager);
    int add_replace_frame(int frame_id);
    void lock_range(int start_frame, int end_frame);
    void unlock_range(int start_frame, int end_frame);

    /**
     * @brief Make a private copy of the data from the ring buffer
     *
     * @param event_id unique identifier of the event in the FRB pipeline
     * @param trigger_start_fpga start time, or -1 to use the earliest data available
     * @param trigger_length_fpga number of FPGA samples to include in the dump
     *
     * @return A fully initialized `basebandDumpData` if the call succeeded, or
     * an empty one if the frame data was not availabe for the time requested
     */
    kotekan::basebandDumpData::Status get_data(uint64_t event_id, int64_t trigger_start_fpga,
                                               int64_t trigger_length_fpga);

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& readout_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& readout_in_progress_metric;
};

#endif
