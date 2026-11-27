#include "BasebandWriter.hpp"

#include "BasebandFrameView.hpp" // for BasebandFrameView
#include "BasebandMetadata.hpp"  // for BasebandMetadata
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp"    // for DEBUG, INFO, ERROR, FATAL_ERROR, WARN
#include "visUtil.hpp"           // for current_time, frameID, modulo, movingAverage

#include "fmt.hpp" // for format

#include <algorithm>  // for max
#include <atomic>     // for atomic_bool
#include <chrono>     // for duration, operator-, seconds, operator/, operator>, tim...
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function, ref
#include <math.h>     // for round
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <sys/stat.h> // for mkdir, stat, S_ISDIR, S_IRGRP, S_IROTH, S_IRWXU, S_IXGRP, S_IXOTH
#include <thread>     // for sleep_for, thread
#include <tuple>      // for forward_as_tuple
#include <utility>    // for pair, piecewise_construct, move
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(BasebandWriter);

BasebandWriter::BasebandWriterDestination::BasebandWriterDestination(const std::string& file_name,
                                                                     const uint32_t& frame_size) :
    file(file_name, frame_size),
    last_updated(current_time()) {}


BasebandWriter::BasebandWriter(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&BasebandWriter::main_thread, this)),

    _root_path(config.get_default<std::string>(unique_name, "root_path", ".")),
    _dump_timeout(config.get_default<double>(unique_name, "dump_timeout", 60)),
    _max_frames_per_second(config.get_default<double>(unique_name, "max_frames_per_second", 0)),
    _num_threads(config.get_default<uint32_t>(unique_name, "num_threads", 1)),
    _frame_size(config.get<uint32_t>(unique_name, "samples_per_data_set")
                    * config.get<uint32_t>(unique_name, "num_elements")
                + sizeof(BasebandMetadata)),
    in_buf(get_buffer("in_buf")), write_in_progress_metric(Metrics::instance().add_gauge(
                                      "kotekan_baseband_writeout_in_progress", unique_name)),
    active_event_dumps_metric(
        Metrics::instance().add_gauge("kotekan_baseband_writeout_active_events", unique_name)),
    write_time_metric(
        Metrics::instance().add_gauge("kotekan_writer_write_time_seconds", unique_name)),
    bytes_written_metric(
        Metrics::instance().add_counter("kotekan_writer_bytes_total", unique_name)) {
    if (_num_threads < 1 || _num_threads > MAX_CONSUMERS) {
        throw std::runtime_error(fmt::format("BasebandWriter {:s}: num_threads ({:d}) must be in "
                                             "[1, {:d}]",
                                             unique_name, _num_threads, MAX_CONSUMERS));
    }
    // Each writer thread is its own consumer of `in_buf`. A single thread keeps the stage's
    // name, so the default configuration registers exactly as before.
    for (uint32_t i = 0; i < _num_threads; ++i) {
        auto writer = std::make_unique<WriterThread>();
        writer->index = i;
        writer->consumer_name =
            _num_threads == 1 ? unique_name : fmt::format("{:s}/thread_{:d}", unique_name, i);
        register_consumer(in_buf, writer->consumer_name.c_str());
        writers.push_back(std::move(writer));
    }
}


void BasebandWriter::main_thread() {
    // The writer threads inherit this thread's CPU affinity, so apply it before starting them
    apply_cpu_affinity();

    std::thread closing_thread(&BasebandWriter::close_old_events, this);

    std::vector<std::thread> threads;
    for (auto& writer : writers) {
        threads.emplace_back(&BasebandWriter::writer_loop, this, std::ref(*writer));
    }
    for (auto& thread : threads) {
        thread.join();
    }

    {
        std::lock_guard<std::mutex> lk(closing_mtx);
        closing_stop = true;
    }
    stop_closing.notify_one();
    closing_thread.join();
}


void BasebandWriter::writer_loop(WriterThread& writer) {
    frameID frame_id(in_buf);
    const char* consumer_name = writer.consumer_name.c_str();

    // Each thread throttles to its share of the stage's maximum throughput
    const double max_frames_per_second = _max_frames_per_second / _num_threads;
    auto period_start = std::chrono::steady_clock::now();
    unsigned int frames_in_period = 0;

    try {
        while (!stop_thread) {
            // Wait for the buffer to be filled with data
            if (wait_for_full_frame(in_buf, consumer_name, frame_id) == nullptr) {
                break;
            }

            // Pass over the frequencies the other writer threads write
            if (_num_threads > 1
                && BasebandFrameView(in_buf, frame_id).metadata()->freq_id % _num_threads
                       != writer.index) {
                mark_frame_empty(in_buf, consumer_name, frame_id++);
                continue;
            }

            // Write the frame to its event+frequency destination file
            write_data(writer, frame_id);

            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> diff = now - period_start;
            if (diff > std::chrono::seconds(1)) {
                DEBUG("Restart step count ({:.3f} s)", diff.count());
                period_start = now;
                frames_in_period = 1;
            } else {
                ++frames_in_period;

                double period_throughput = frames_in_period / diff.count();
                DEBUG("Current step count: {} ({:.2f})", frames_in_period, period_throughput);
                if (max_frames_per_second > 0 && period_throughput > max_frames_per_second) {
                    WARN("Throughput exceeded: {:.2f} >> {}", period_throughput,
                         max_frames_per_second);
                    const std::chrono::duration<double> remaining_in_period =
                        std::chrono::seconds(1) - diff;
                    if (frames_in_period >= max_frames_per_second) {
                        INFO("Sleep until the end of the period: {:.3}s",
                             remaining_in_period.count());
                    } else {
                        const auto throttle_rate =
                            remaining_in_period / (max_frames_per_second - frames_in_period);
                        INFO("Over the quota after {} frames; ({}s remaining, throttle by {:.3} "
                             "s)",
                             frames_in_period, remaining_in_period.count(), throttle_rate.count());
                        std::this_thread::sleep_for(throttle_rate);
                    }
                }
            }

            // Mark the buffer and move on
            mark_frame_empty(in_buf, consumer_name, frame_id++);
        }
    } catch (std::exception& e) {
        FATAL_ERROR("BasebandWriter {:s} stopped: {:s}", writer.consumer_name, e.what());
    }
}

void BasebandWriter::write_data(WriterThread& writer, int frame_id) {
    try {
        const auto frame = BasebandFrameView(in_buf, frame_id);
        const auto metadata = frame.metadata();
        const auto event_id = metadata->event_id;
        const auto freq_id = metadata->freq_id;
        DEBUG("Writing frame {} for event_id: {}, freq_id: {}", metadata->frame_fpga_seq,
              metadata->event_id, metadata->freq_id);

        {
            std::lock_guard<std::mutex> stats_lk(stats_mtx);
            write_in_progress_metric.set(++writes_in_progress > 0);
        }

        // Lock this thread's event->freq->file map
        std::unique_lock lk(writer.mtx);

        const std::string event_directory_name =
            fmt::format("{:s}/baseband_raw_{:d}", _root_path, event_id);
        DEBUG("Event directory: {:s}", event_directory_name);

        if (writer.baseband_events.count(event_id) == 0) {
            DEBUG("Creating event directory: {:s}", event_directory_name);
            const int mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
            const int result = mkdir(event_directory_name.c_str(), mode);
            const int errsv = errno;
            // Another writer thread (or an earlier run) may have created it already. That is
            // fine as long as it is a directory.
            struct stat st;
            const bool existing_dir = result != 0 && errsv == EEXIST
                                      && stat(event_directory_name.c_str(), &st) == 0
                                      && S_ISDIR(st.st_mode);
            if (result != 0 && !existing_dir) {
                ERROR("Failed to create event directory: {:s}", event_directory_name);
                switch (errsv) {
                    case EACCES:
                        ERROR("Permission denied: {:s}", std::strerror(errsv));
                        throw std::runtime_error(std::string("Permission denied: ")
                                                 + std::strerror(errsv));
                    case EEXIST:
                        ERROR("Path exists but is not a directory: {:s}", event_directory_name);
                        throw std::runtime_error(std::string("Path exists but is not a directory: ")
                                                 + event_directory_name);
                    case EINVAL:
                        ERROR("Invalid argument: {:s}", std::strerror(errsv));
                        throw std::runtime_error(std::string("Invalid argument: ")
                                                 + std::strerror(errsv));
                    case ELOOP:
                        ERROR("Too many symbolic links encountered: {:s}", std::strerror(errsv));
                        throw std::runtime_error(
                            std::string("Too many symbolic links encountered: ")
                            + std::strerror(errsv));
                    case ENAMETOOLONG:
                        ERROR("Path name too long: {:s}", std::strerror(errsv));
                        throw std::runtime_error(std::string("Path name too long: ")
                                                 + std::strerror(errsv));
                    case ENOENT:
                        ERROR("No such file or directory: {:s}", std::strerror(errsv));
                        throw std::runtime_error(std::string("No such file or directory: ")
                                                 + std::strerror(errsv));
                    case ENOTDIR:
                        ERROR("A component of the path is not a directory: {:s}",
                              std::strerror(errsv));
                        throw std::runtime_error(
                            std::string("A component of the path is not a directory: ")
                            + std::strerror(errsv));
                    case EPERM:
                        ERROR("Operation not permitted: {:s}", std::strerror(errsv));
                        throw std::runtime_error(std::string("Operation not permitted: ")
                                                 + std::strerror(errsv));
                    case EROFS:
                        ERROR("Read-only file system: {:s}", std::strerror(errsv));
                        throw std::runtime_error(std::string("Read-only file system: ")
                                                 + std::strerror(errsv));
                    default:
                        ERROR("Unable to create directory: {:s}", std::strerror(errsv));
                        throw std::runtime_error(std::string("Unable to create directory: ")
                                                 + std::strerror(errsv));
                }
            }
            DEBUG("Completed creating event directory: {:s}", event_directory_name);

            std::lock_guard<std::mutex> stats_lk(stats_mtx);
            ++active_events[event_id];
            active_event_dumps_metric.set(active_events.size());
        }

        const std::string file_name =
            fmt::format("{:s}/baseband_{:d}_{:d}", event_directory_name, event_id, freq_id);
        DEBUG("Output filename: {:s}", file_name);

        if (writer.baseband_events[event_id].count(freq_id) == 0) {
            // NOTE: emplace the file instance or it will get closed by the destructor
            writer.baseband_events[event_id].emplace(std::piecewise_construct,
                                                     std::forward_as_tuple(freq_id),
                                                     std::forward_as_tuple(file_name, _frame_size));
        }
        auto& freq_dump_destination = writer.baseband_events[event_id].at(freq_id);
        BasebandFileRaw& baseband_file = freq_dump_destination.file;

        const double start = current_time();
        int32_t write_status = baseband_file.write_frame({in_buf, frame_id});
        freq_dump_destination.last_updated = current_time();
        const double elapsed = freq_dump_destination.last_updated - start;

        {
            std::lock_guard<std::mutex> stats_lk(stats_mtx);
            write_in_progress_metric.set(--writes_in_progress > 0);
        }

        if (write_status == -1) {
            ERROR("Output file is corrupt, dropping frame going to {:s}", file_name);
        } else if (write_status == 0) {
            FATAL_ERROR("Cannot write to file {:s}, maybe out of disk space?", file_name);
        } else {
            DEBUG("Written frame with event id {:d} and freq {:d} to {:s}", event_id, freq_id,
                  file_name);

            bytes_written_metric.inc(frame.data_size() + sizeof(BasebandMetadata));

            // Update average write time in prometheus
            std::lock_guard<std::mutex> stats_lk(stats_mtx);
            write_time.add_sample(elapsed);
            write_time_metric.set(write_time.average());
        }
    } catch (std::exception& e) {
        ERROR("Exception in BasebandWriter: {:s}", e.what());
        throw;
    } catch (...) {
        ERROR("Unknown exception in BasebandWriter");
        throw;
    }
    DEBUG("Finished writing data to file");
}


void BasebandWriter::close_old_events() {
    DEBUG("Starting the file-closing thread");
    // Do not run the loop more often than once a minute
    const int sweep_cadence_s = std::max(60.0, round(_dump_timeout / 2));
    while (true) {
        double now = current_time();
        DEBUG("Run file-closing thread {:.1f}", now);
        {
            std::unique_lock<std::mutex> lk(closing_mtx);
            if (stop_closing.wait_for(lk, std::chrono::seconds(sweep_cadence_s),
                                      [this]() { return closing_stop; })) {
                break;
            }
        }

        // Otherwise, we've waited long enough and can do the sweep, one writer thread at a time
        for (auto& writer : writers) {
            std::lock_guard<std::mutex> lk(writer->mtx);
            auto& baseband_events = writer->baseband_events;
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
                    std::lock_guard<std::mutex> stats_lk(stats_mtx);
                    if (--active_events[event_it->first] <= 0) {
                        active_events.erase(event_it->first);
                    }
                    active_event_dumps_metric.set(active_events.size());
                    event_it = baseband_events.erase(event_it);
                } else {
                    ++event_it;
                }
            }
        }
    }
    DEBUG("File-closing thread done");
}
