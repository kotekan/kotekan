#ifndef APPLY_GAINS_HPP
#define APPLY_GAINS_HPP

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "updateQueue.hpp"
#include "util.h"
#include "visFile.hpp"
#include "visUtil.hpp"

#include <shared_mutex>
#include <unistd.h>


/**
 * @class applyGains
 * @brief Receives gains and apply them to the output buffer.
 *
 * This process registers as a subscriber to an updatable config block. The
 * full name of the block should be defined in the value <updatable_block>
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
 * will provide new flagging values (e.g. "/dynamic_block/gains").
 * @conf   gains_dir        String. The path to the directory holding the gains
 * file.
 * @conf   tcombine         Double. Time (in seconds) over which to combine old
 * and new gains to prevent discontinuities. Default is 5 minutes.
 * @conf   num_kept_updates Int.    The number of gain updates stored in a FIFO.
 * @conf   num_threads      Int.    Number of threads to run. Default is 1.
 *
 * @par Metrics
 * @metric kotekan_applygains_late_update_count The number of updates received
 *     too late (The start time of the update is older than the currently
 *     processed frame).
 * @metric kotekan_applygains_late_frame_count The number of frames received
 *     late (The frames timestamp is older then all start times of stored
 *     updates).
 * @metric kotekan_applygains_update_age_seconds The time difference in
 *     seconds between the current frame being processed and the time stamp of
 *     the gains update being applied.
 *
 * @author Mateus Fandino
 */
class applyGains : public KotekanProcess {

public:
    struct gainUpdate {
        std::vector<std::vector<cfloat>> gain;
        std::vector<std::vector<float>> weight;
    };

    /// Default constructor
    applyGains(Config& config, const string& unique_name, bufferContainer& buffer_container);

    /// Main loop for the process
    void main_thread() override;

    /// Callback function to receive updates on timestamps from configUpdater
    bool receive_update(nlohmann::json& json);

    /// Check if file to read exists
    bool fexists(const std::string& filename);

private:
    // Parameters saved from the config files

    /// Path to gains directory
    std::string gains_dir;

    /// Number of gains updates to maintain
    uint64_t num_kept_updates;

    /// Time over which to blend old and new gains in seconds. Default is 5 minutes.
    double tcombine;

    /// The gains and when to start applying them in a FIFO (len set by config)
    updateQueue<gainUpdate> gains_fifo;

    /// Output buffer with gains applied
    Buffer* out_buf;
    /// Input buffer to read from
    Buffer* in_buf;

    /// Mutex to protect access to gains
    // N.B. `shared_mutex` is only available in C++17
    std::shared_timed_mutex gain_mtx;

    /// Timestamp of the current frame
    std::atomic<timespec> ts_frame{{0, 0}};

    /// Number of updates received too late
    std::atomic<size_t> num_late_updates;

    /// Number of frames received too late, every thread must be able to increment
    std::atomic<size_t> num_late_frames;

    /// Entrancepoint for n threads. Each thread takes frames with a
    /// different frame_id from the buffer and applies gains.
    void apply_thread(int thread_id);

    /// Vector to hold the thread handles
    std::vector<std::thread> thread_handles;

    /// Number of parallel threads accessing the same buffers (default 1)
    uint32_t num_threads;
};


#endif
