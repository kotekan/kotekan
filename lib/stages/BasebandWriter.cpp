#include "BasebandWriter.hpp"

#include "StageFactory.hpp"
#include "visUtil.hpp" // for current_time

#include <chrono>     // for seconds
#include <errno.h>    // for errno
#include <fcntl.h>    // for O_CREAT, O_WRONLY
#include <sys/stat.h> // for mkdir
#include <thread>     // for thread

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
    in_buf(get_buffer("in_buf")),
    write_time_metric(
        Metrics::instance().add_gauge("kotekan_writer_write_time_seconds", unique_name)) {
    register_consumer(in_buf, unique_name.c_str());
}


void BasebandWriter::main_thread() {
    frameID frame_id(in_buf);
    INFO("Start the closing thread");
    std::thread closing_thread(&BasebandWriter::close_old_events, this);

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Write the frame to its event+frequency destination file
        write_data(in_buf, frame_id);

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);
    }

    stop_closing.notify_one();
    closing_thread.join();
}


void BasebandWriter::write_data(Buffer* in_buf, int frame_id) {
    const auto frame = BasebandFrameView(in_buf, frame_id);
    const auto metadata = frame.metadata();

    INFO("Frame {} from {}/{}", metadata->fpga_seq, metadata->event_id, metadata->freq_id);

    // Lock the event->freq->file map
    std::unique_lock lk(mtx);
    const std::string event_directory_name =
        fmt::format("{:s}/baseband_raw_{:d}", _root_path, metadata->event_id);
    if (baseband_events.count(metadata->event_id) == 0) {
        mkdir(event_directory_name.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }

    const std::string file_name = fmt::format("{:s}/baseband_{:d}_{:d}", event_directory_name,
                                              metadata->event_id, metadata->freq_id);
    if (baseband_events[metadata->event_id].count(metadata->freq_id) == 0) {
        // NOTE: emplace the file instance or it will get closed by the destructor
        baseband_events[metadata->event_id].emplace(metadata->freq_id, file_name);
    }
    auto& freq_dump_destination = baseband_events[metadata->event_id].at(metadata->freq_id);
    BasebandFileRaw& baseband_file = freq_dump_destination.file;

    const double start = current_time();
    ssize_t bytes_written = baseband_file.write_frame({in_buf, frame_id});
    freq_dump_destination.last_updated = current_time();
    const double elapsed = freq_dump_destination.last_updated - start;

    if (bytes_written != metadata->valid_to) {
        ERROR("Failed to write buffer to disk for file {:s}", file_name);
        exit(-1);
    }
    if (bytes_written < in_buf->frame_size) {
        INFO("Closing {}/{}", metadata->event_id, metadata->freq_id);
        baseband_events[metadata->event_id].erase(
            baseband_events[metadata->event_id].find(metadata->freq_id));
    }

    // Update average write time in prometheus
    write_time.add_sample(elapsed);
    write_time_metric.set(write_time.average());
}


void BasebandWriter::close_old_events() {
    // Do not run the loop more often than once a minute
    const int sweep_cadence_s = std::max(60.0, round(_dump_timeout / 2));
    while (!stop_thread) {
        double now = current_time();
        DEBUG("Run closing thread {:.1f}", now);
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
                    DEBUG("Cleaning up {}", event_freq->second.file.name);
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
    }
    INFO("Closing thread done");
}
