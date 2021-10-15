#include "BasebandWriter.hpp"

#include "BasebandFrameView.hpp" // for BasebandFrameView
#include "BasebandMetadata.hpp"  // for BasebandMetadata
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp"    // for DEBUG, INFO, ERROR, WARN
#include "visUtil.hpp"           // for current_time, frameID, modulo, movingAverage

#include "fmt.hpp" // for format

#include <algorithm>   // for max
#include <atomic>      // for atomic_bool
#include <chrono>      // for duration, operator-, seconds, operator/, operator>, tim...
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <math.h>      // for round
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdlib.h>    // for exit
#include <sys/stat.h>  // for mkdir, S_IRGRP, S_IROTH, S_IRWXU, S_IXGRP, S_IXOTH
#include <sys/types.h> // for ssize_t
#include <thread>      // for sleep_for, thread
#include <utility>     // for pair
#include <vector>      // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(BasebandWriter);

BasebandWriter::BasebandWriterDestination::BasebandWriterDestination(const std::string& file_name) :
    file(file_name),
    last_updated(current_time()) {}


BasebandWriter::BasebandWriter(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&BasebandWriter::main_thread, this)),

    _root_path(config.get_default<std::string>(unique_name, "root_path", ".")),
    _dump_timeout(config.get_default<double>(unique_name, "dump_timeout", 60)),
    _max_frames_per_second(config.get_default<double>(unique_name, "max_frames_per_second", 0)),
    in_buf(get_buffer("in_buf")),
    write_in_progress_metric(
        Metrics::instance().add_gauge("kotekan_baseband_writeout_in_progress", unique_name)),
    active_event_dumps_metric(
        Metrics::instance().add_gauge("kotekan_baseband_writeout_active_events", unique_name)),
    write_time_metric(
        Metrics::instance().add_gauge("kotekan_writer_write_time_seconds", unique_name)),
    bytes_written_metric(
        Metrics::instance().add_counter("kotekan_writer_bytes_total", unique_name)) {
    register_consumer(in_buf, unique_name.c_str());
}


void BasebandWriter::main_thread() {
    frameID frame_id(in_buf);
    std::thread closing_thread(&BasebandWriter::close_old_events, this);

    std::chrono::time_point<std::chrono::steady_clock> period_start;
    unsigned int frames_in_period = 0;

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Write the frame to its event+frequency destination file
        write_data(in_buf, frame_id);

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff = now - period_start;
        if (diff > std::chrono::seconds(1)) {
            DEBUG("Restart step count ({:.3f} s)", diff);
            period_start = now;
            frames_in_period = 1;
        } else {
            ++frames_in_period;

            double period_throughput = frames_in_period / diff.count();
            DEBUG("Current step count: {} ({:.2f})", frames_in_period, period_throughput);
            if (_max_frames_per_second > 0 && period_throughput > _max_frames_per_second) {
                WARN("Throughput exceeded: {:.2f} >> {}", period_throughput,
                     _max_frames_per_second);
                const std::chrono::duration<double> remaining_in_period =
                    std::chrono::seconds(1) - diff;
                if (frames_in_period >= _max_frames_per_second) {
                    INFO("Sleep until the end of the period: {:.3}s", remaining_in_period.count());
                } else {
                    const auto throttle_rate =
                        remaining_in_period / (_max_frames_per_second - frames_in_period);
                    INFO("Over the quota after {} frames; ({}s remaining, throttle by {:.3} s)",
                         frames_in_period, remaining_in_period.count(), throttle_rate.count());
                    std::this_thread::sleep_for(throttle_rate);
                }
            }
        }

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);
    }

    stop_closing.notify_one();
    closing_thread.join();
}


void BasebandWriter::write_data(Buffer* in_buf, int frame_id) {
    const auto frame = BasebandFrameView(in_buf, frame_id);
    const auto metadata = frame.metadata();

    const auto event_id = metadata->event_id;
    const auto freq_id = metadata->freq_id;
    DEBUG("Frame {} from {}/{}", metadata->frame_fpga_seq, event_id, freq_id);

    write_in_progress_metric.set(1);

    // Lock the event->freq->file map
    std::unique_lock lk(mtx);
    active_event_dumps_metric.set(baseband_events.size());

    const std::string event_directory_name =
        fmt::format("{:s}/baseband_raw_{:d}", _root_path, event_id);
    if (baseband_events.count(event_id) == 0) {
        mkdir(event_directory_name.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }

    const std::string file_name =
        fmt::format("{:s}/baseband_{:d}_{:d}", event_directory_name, event_id, freq_id);
    if (baseband_events[event_id].count(freq_id) == 0) {
        // NOTE: emplace the file instance or it will get closed by the destructor
        baseband_events[event_id].emplace(freq_id, file_name);
    }
    auto& freq_dump_destination = baseband_events[event_id].at(freq_id);
    BasebandFileRaw& baseband_file = freq_dump_destination.file;

    const double start = current_time();
    ssize_t bytes_written = baseband_file.write_frame({in_buf, frame_id});
    freq_dump_destination.last_updated = current_time();
    const double elapsed = freq_dump_destination.last_updated - start;

    write_in_progress_metric.set(0);

    if (bytes_written != in_buf->frame_size) {
        ERROR("Failed to write buffer to disk for file {:s}", file_name);
        exit(-1);
    } else {
        DEBUG("Written {} bytes of data to {:s}", bytes_written, file_name);
    }
    bytes_written_metric.inc(bytes_written);

    // Update average write time in prometheus
    write_time.add_sample(elapsed);
    write_time_metric.set(write_time.average());
}


void BasebandWriter::close_old_events() {
    DEBUG("Starting the file-closing thread");
    // Do not run the loop more often than once a minute
    const int sweep_cadence_s = std::max(60.0, round(_dump_timeout / 2));
    while (!stop_thread) {
        double now = current_time();
        DEBUG("Run file-closing thread {:.1f}", now);
        std::unique_lock lk(mtx);
        if (stop_closing.wait_for(lk, std::chrono::seconds(sweep_cadence_s))
            != std::cv_status::timeout) {
            // is it a notification to exit or a spurious interrupt?
            if (stop_thread) {
                return;
            } else {
                continue;
            }
        }

        // Otherwise, we've waited long enough and can do the sweep
        for (auto event_it = baseband_events.begin(); event_it != baseband_events.end();) {
            for (auto event_freq = event_it->second.begin();
                 event_freq != event_it->second.end();) {
                // close the frequency file that's been inactive for over a minute
                if (now - event_freq->second.last_updated > _dump_timeout) {
                    INFO("Closing {}", event_freq->second.file.name);
                    event_freq = event_it->second.erase(event_freq);
                } else {
                    ++event_freq;
                }
            }
            if (event_it->second.empty()) {
                event_it = baseband_events.erase(event_it);
            } else {
                ++event_it;
            }
        }
        active_event_dumps_metric.set(baseband_events.size());
    }
    DEBUG("File-closing thread done");
}
