/*****************************************
@file
@brief Receive and set flags for the visibility data.
- ReceiveFlags : public kotekan::Stage
*****************************************/
#ifndef RECEIVEFLAGS_H
#define RECEIVEFLAGS_H

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, state_id_t
#include "prometheusMetrics.hpp" // for Counter, Gauge
#include "updateQueue.hpp"       // for updateQueue
#include "visBuffer.hpp"         // for VisFrameView

#include "json.hpp" // for json

#include <map>      // for map
#include <mutex>    // for mutex
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <time.h>   // for size_t, timespec
#include <utility>  // for pair
#include <vector>   // for vector

/**
 * @class ReceiveFlags
 * @brief Receives input flags and adds them to the output buffer.
 *
 * This stage registeres as a subscriber to an updatable config block. The
 * full name of the block should be defined in the value \<updatable_block\>
 *
 * @note If there are no other consumers on the input buffer it will be able to
 *       do a much faster zero copy transfer of the frame from input to output
 *       buffer.
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format VisBuffer.
 *         @buffer_metadata VisMetadata
 * @buffer out_buf The output stream.
 *         @buffer_format VisBuffer.
 *         @buffer_metadata VisMetadata
 *
 * @conf   num_elements     Int.    The number of elements (i.e. inputs) in the
 *   correlator data.
 * @conf   updatable_block  String. The full name of the updatable_block that
 *   will provide new flagging values (e.g. "/dynamic_block/flagging").
 *
 * @par Metrics
 * @metric kotekan_receiveflags_late_update_count The number of updates received
 *     too late (The start time of the update is older than the currently
 *     processed frame).
 * @metric kotekan_receiveflags_late_frame_count The number of frames received
 *     late (The frames timestamp is older then all start times of stored
 *     updates).
 * @metric kotekan_receiveflags_update_age_seconds The time difference in
 *   seconds between the current frame being processed and the time stamp of
 *   the flag update being applied.
 *
 * @author Rick Nitsche
 */
class ReceiveFlags : public kotekan::Stage {
public:
    /// Constructor
    ReceiveFlags(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Main loop, saves flags in the frames
    void main_thread() override;

    /// This will be called by configUpdater
    bool flags_callback(nlohmann::json& json);

private:
    /// Copy the freshest flags into the frame or return false if no valid update for
    /// this frame is available
    bool copy_flags_into_frame(const VisFrameView& frame_out);

    // this is faster than std::queue/deque
    /// The bad_input chan_id's and when to start applying them in a FIFO
    /// (len set by config)
    updateQueue<std::pair<state_id_t, std::vector<float>>> flags;

    // Map from the state being applied and input dataset to the output dataset.
    // This is used to keep track of the labels we should be applying for
    // timesamples coming out of order.
    std::map<std::pair<state_id_t, dset_id_t>, dset_id_t> output_dataset_ids;

    /// Input buffer
    Buffer* buf_in;

    /// Output buffer
    Buffer* buf_out;

    /// To make sure flags are not modified and saved at the same time
    std::mutex flags_lock;

    /// Timestamp of the current frame
    timespec ts_frame = {0, 0};

    /// Number of frames received late
    kotekan::prometheus::Counter& receiveflags_late_frame_counter;

    /// Update ages
    kotekan::prometheus::Gauge& receiveflags_update_age_metric;

    /// Number of updates received too late
    kotekan::prometheus::Counter& late_updates_counter;

    // config values
    /// Number of elements
    size_t num_elements;

    /// Number of updates to keep track of
    uint32_t num_kept_updates;
};

#endif /* RECEIVEFLAGS_H */
