/*****************************************
@file
@brief Baseband BasebandWriter stage.
- BasebandWriter : public
*****************************************/
#ifndef BASEBAND_WRITER_HPP
#define BASEBAND_WRITER_HPP

#include "BasebandFileRaw.hpp"   // for BasebandFileRaw
#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Gauge, Counter
#include "visUtil.hpp"           // for movingAverage

#include <condition_variable> // for condition_variable
#include <cstdint>            // for uint32_t, uint64_t
#include <mutex>              // for mutex
#include <string>             // for string
#include <unordered_map>      // for unordered_map

/**
 * @class BasebandWriter
 * @brief Writes raw baseband event frames to disk.
 *
 * Streams `BasebandBuffer` frames from a single input buffer and spools per-event, per-frequency
 * raw files on disk. Each file is named `baseband_<event>_<freq>` inside a directory
 * `baseband_raw_<event>` under `root_path`, and stale files are closed after `dump_timeout`
 * seconds of inactivity. Frame rate can optionally be throttled by `max_frames_per_second`; when
 * set the stage will sleep to keep the moving throughput under that cap. A background thread
 * periodically cleans up old events while the main thread writes frames and updates metrics.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *         @buffer_format BasebandBuffer structured
 *         @buffer_metadata BasebandMetadata
 *
 * @conf root_path             String, default ".". Base directory for output.
 * @conf samples_per_data_set  UInt. Samples in each frame (used to size payload).
 * @conf num_elements          UInt. Number of elements (used to size payload).
 * @conf dump_timeout          Double, default 60. Seconds of inactivity before closing a file.
 * @conf max_frames_per_second Double, default 0. If >0 limit read/write rate to this FPS.
 *
 * @par Metrics
 * @metric kotekan_baseband_writeout_in_progress
 *         Set to 1 when a frequency is being written to, 0 otherwise.
 *
 * @metric kotekan_baseband_writeout_active_events
 *         The number of events with any raw files still open
 *
 * @metric kotekan_writer_write_time_seconds
 *         The write time of the raw writer. An exponential moving average over ~10
 *         samples.
 *
 * @metric kotekan_writer_bytes_total
 *         Number of bytes written to files since the start of this stage
 *
 * @par Example
 * @code
 * baseband_writer:
 *   kotekan_stage: BasebandWriter
 *   in_buf: baseband_in
 *   root_path: /data/baseband
 *   samples_per_data_set: 49152
 *   num_elements: 2048
 *   dump_timeout: 120
 *   max_frames_per_second: 0   # no throttling
 * @endcode
 *
 */
class BasebandWriter : public kotekan::Stage {
public:
    BasebandWriter(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    void main_thread() override;

private:
    /**
     * @brief write a frame of data into a baseband dump file
     *
     * @param in_buf   The buffer the frame is in.
     * @param frame_id The id of the frame to write.
     */
    void write_data(Buffer* in_buf, int frame_id);

    /**
     * @brief runs the loop to periodically close stale dump files
     */
    void close_old_events();

    // Parameters saved from the config file
    std::string _root_path;
    double _dump_timeout;
    double _max_frames_per_second;
    uint32_t _frame_size;

    /// Input buffer to read from
    Buffer* in_buf;

    // Convenience class just so we can have a pair of raw file and time with a single "string"
    // constructor that can be used from `map::emplace`
    class BasebandWriterDestination {
    public:
        BasebandWriterDestination(const std::string&, const uint32_t&);
        BasebandFileRaw file;
        double last_updated;
    };
    /// The set of active baseband dump files, keyed by their event id to
    /// frequency map
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, BasebandWriterDestination>>
        baseband_events;

    /// synchronizes access to the event map
    std::mutex mtx;

    /// notifies the file-closing thread (i.e., running `close_old_events`)
    std::condition_variable stop_closing;

    /// Keep track of the average write time
    movingAverage write_time;

    // Prometheus metric to indicate when a per-frequency writeout is in progress
    kotekan::prometheus::Gauge& write_in_progress_metric;

    // Prometheus metric that counts the number of event dumps with files still open
    kotekan::prometheus::Gauge& active_event_dumps_metric;

    // Prometheus metric to expose the value of `write_time`
    kotekan::prometheus::Gauge& write_time_metric;

    // Prometheus counter of total bytes written by the stage
    kotekan::prometheus::Counter& bytes_written_metric;
};

#endif // BASEBAND_WRITER_HPP
