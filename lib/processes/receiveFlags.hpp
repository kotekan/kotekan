/*****************************************
@file
@brief Receive and set flags for the visibility data.
- receiveFlags : public kotekan::Stage
*****************************************/
#ifndef RECEIVEFLAGS_H
#define RECEIVEFLAGS_H

#include "KotekanProcess.hpp"
#include "updateQueue.hpp"

#include <mutex>

/**
 * @class receiveFlags
 * @brief Receives input flags and adds them to the output buffer.
 *
 * This process registeres as a subscriber to an updatable config block. The
 * full name of the block should be defined in the value <updatable_block>
 *
 * @note If there are no other consumers on the input buffer it will be able to
 *       do a much faster zero copy transfer of the frame from input to output
 *       buffer.
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata
 * @buffer out_buf The output stream.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata
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
class receiveFlags : public kotekan::Stage {
public:
    /// Constructor
    receiveFlags(kotekan::Config& config, const string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    /// Main loop, saves flags in the frames
    void main_thread() override;

    /// This will be called by configUpdater
    bool flags_callback(nlohmann::json& json);

private:
    // this is faster than std::queue/deque
    /// The bad_input chan_id's and when to start applying them in a FIFO
    /// (len set by config)
    updateQueue<std::vector<float>> flags;

    /// Input buffer
    Buffer* buf_in;

    /// Output buffer
    Buffer* buf_out;

    /// To make sure flags are not modified and saved at the same time
    std::mutex flags_lock;

    /// Timestamp of the current frame
    timespec ts_frame = {0, 0};

    /// Number of updates received too late
    size_t num_late_updates;

    // config values
    /// Number of elements
    size_t num_elements;

    /// Number of updates to keept track of
    uint32_t num_kept_updates;
};

#endif /* RECEIVEFLAGS_H */
