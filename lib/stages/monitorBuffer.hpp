/**
 * @file
 * @brief Stage to watch a buffer or buffers and exit the system if it doesn't
 *        get new data within a set timeout
 *  - monitorBuffer : public kotekan::Stage
 */

#ifndef MONITOR_BUFFER_H
#define MONITOR_BUFFER_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t, uint64_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class monitorBuffer
 * @brief Watches a set of buffers to make sure they are getting data
 *        within a given timeout, otherwise exits.
 *
 * @note This will not detect buffers which never get data, the buffer
 *       must get at least one frame before this system will start checking.
 *
 * @par Buffers
 * @buffer bufs An array of kotekan buffers
 *  @buffer_format Any
 *  @buffer_metadata Any
 *
 * @conf timeout            Int, default 60. Time to wait in seconds for new frame
 *                          before exiting.   Must be > 1
 * @conf fill_threshold     Float, default 2.0 (disabled)  The ratio of full to total frames,
 *                          which if exceeded with trigger an exit.
 * @conf graceful_shutdown  Bool, default false. If true, instead of fatal error on timeout,
 *                          send shutdown signals to all stages and wait for them to exit.
 * @conf clean_exit_after_frames Int, default -1. If > 0 the stage registers as a lightweight
 *                          consumer, counts frames on each buffer, and triggers a clean exit
 *                          once every monitored buffer has seen at least this many frames.
 */
class monitorBuffer : public kotekan::Stage {
public:
    /// Common constructor
    monitorBuffer(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);

    /// Destructor
    virtual ~monitorBuffer();

    /// Watches the list of buffers for timeouts
    void main_thread() override;

private:
    /// Internal list of buffers to check
    std::vector<Buffer*> buffers;

    /// The timeout after which kotekan exits
    int timeout;

    /// The maximum fraction of full buffers before the system exits
    /// If set above 1 then this check is disabled
    float fill_threshold;

    /// Perform a graceful shutdown instead of fatal error on timeout
    bool graceful_shutdown;

    /// Whether to wait for data from the first frame before starting monitoring
    bool wait_for_first_frame;

    /// Whether to stop monitoring after this number of frames have been seen
    int clean_exit_after_frames;

    /// Whether this stage has registered as a buffer consumer in order to count frames
    bool consume_frames = false;

    struct buffer_state {
        Buffer* buffer;
        uint32_t next_frame_id;
        uint64_t frames_seen;
    };

    /// Per-buffer tracking, used when acting as a consumer
    std::vector<buffer_state> buffer_states;
};

#endif
