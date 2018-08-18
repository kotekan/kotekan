#ifndef RECEIVEFLAGS_H
#define RECEIVEFLAGS_H

#include "updateQueue.hpp"
#include "gsl-lite.hpp"

#include "KotekanProcess.hpp"
#include <mutex>

/**
 * @class receiveFlags
 * @brief Receives input flags and adds them to the output buffer.
 *
 * This process registeres as a subscriber to an updatable config block. The
 * full name of the block should be defined in the value <updateable_block>
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
 * correlator data.
 * @conf   updatable_block  String. The full name of the updatable_block that
 * will provide new flagging values (e.g. "/dynamic_block/flagging").
 *
 * @metric kotekan_receiveFlags_old_update_seconds  The difference between the
 *  timestamp of a received update and the timestamp of the current frame, in
 *  case the update has a later timestamp than the current frame (in seconds).
 * @metric kotekan_receiveFlags_old_frame_seconds   The difference between the
 *  timestamps of the current frame and the oldes stored update, in case there
 *  is no update with a timestamp that is more recent than the timestamp of the
 *  current frame (in seconds).
 *
 * @author Rick Nitsche
 */
class receiveFlags : public KotekanProcess {
public:

    /// Constructor
    receiveFlags(Config &config, const string& unique_name,
            bufferContainer &buffer_container);

    /// Main loop, saves flags in the frames
    void main_thread() override;

    /// Apply the config from the yaml file
    void apply_config(uint64_t fpga_seq) override;

    /// This will be called by configUpdater
    bool flags_callback(nlohmann::json &json);
private:
    // this is faster than std::queue/deque
    /// The bad_input chan_id's and when to start applying them in a FIFO
    /// (len set by config)
    updateQueue<std::vector<float>> flags;

    /// Input buffer
    Buffer * buf_in;

    /// Output buffer
    Buffer * buf_out;

    /// To make sure flags are not modified and saved at the same time
    std::mutex flags_lock;

    // config values
    /// Number of elements
    size_t num_elems;

    /// Name of the updatable block in conf that contains flags
    std::string updatable_config;

    /// Number of updates to keept track of
    uint32_t num_kept_updates;

    /// Timestamp of the current frame
    timespec ts_frame = {0,0};
};

#endif /* RECEIVEFLAGS_H */

