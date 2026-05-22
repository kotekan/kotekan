#include "monitorBuffer.hpp"

#include <stdint.h>             // for uint32_t, uint64_t, uint8_t
#include <time.h>               // for size_t, timespec
#include <unistd.h>             // for usleep, sleep
#include <fmt/core.h>           // for format, format_string
#include <map>                  // for map, operator!=, _Rb_tree_iterator
#include <utility>              // for pair
#include <algorithm>            // for max

#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer, GenericBuffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "kotekanLogging.hpp"   // for FATAL_ERROR, DEBUG, INFO, WARN
#include "util.h"               // for e_time
#include "visUtil.hpp"          // for double_to_ts
#include "fmt.hpp"              // for compile_string_to_view
#include "errors.h"             // for exit_kotekan, ReturnCode

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_KOTEKAN_STAGE(monitorBuffer);

STAGE_CONSTRUCTOR(monitorBuffer) {

    // Note we do not register as a producer or consumer here.
    buffers = get_buffer_array("bufs");
    // Timeout is in seconds
    timeout = config.get_default<int>(unique_name, "timeout", 60);
    fill_threshold = config.get_default<float>(unique_name, "fill_threshold", 2.0);
    graceful_shutdown = config.get_default<bool>(unique_name, "graceful_shutdown", false);
    wait_for_first_frame = config.get_default<bool>(unique_name, "wait_for_first_frame", true);
    clean_exit_after_frames = config.get_default<int>(unique_name, "clean_exit_after_frames", -1);

    if (clean_exit_after_frames > 0) {
        consume_frames = true;
        buffer_states.reserve(buffers.size());
        for (Buffer* buf : buffers) {
            buf->register_consumer(unique_name);
            buffer_states.push_back({buf, 0u, 0u});
        }
    }

    if (timeout <= 1)
        FATAL_ERROR("monitorBuffer timeout must be > 1 second, got {:d}", timeout);
    if (fill_threshold <= 0.0)
        FATAL_ERROR("monitorBuffer fill_threshold must be > 0.0, got {:f}", fill_threshold);
    if (graceful_shutdown)
        INFO("monitorBuffer configured to perform graceful shutdown on timeout.");
}

monitorBuffer::~monitorBuffer() {
    if (consume_frames) {
        for (Buffer* buf : buffers) {
            buf->unregister_consumer(get_unique_name());
        }
    }
}

void monitorBuffer::main_thread() {

    double monitor_start_time = e_time();
    const std::string& consumer_name = get_unique_name();

    while (!stop_thread) {

        if (consume_frames) {
            for (auto& state : buffer_states) {
                while (!stop_thread) {
                    double deadline = e_time();
                    const timespec deadline_ts = double_to_ts(deadline);
                    int status = state.buffer->wait_for_full_frame_timeout(
                        consumer_name, state.next_frame_id, deadline_ts);
                    if (status == -1) {
                        goto end_loop;
                    }
                    if (status != 0)
                        break;

                    uint8_t* frame =
                        state.buffer->wait_for_full_frame(consumer_name, state.next_frame_id);
                    if (frame == nullptr)
                        goto end_loop;

                    (void)frame;

                    state.buffer->mark_frame_empty(consumer_name, state.next_frame_id);
                    state.frames_seen++;
                    state.next_frame_id = (state.next_frame_id + 1) % state.buffer->num_frames;
                }
            }
        }

        sleep(1);
        double cur_time = e_time();
        for (size_t i = 0; i < buffers.size(); ++i) {
            // skip timeout check for buffer that has reached frame count
            if (consume_frames && clean_exit_after_frames > 0) {
                auto state = buffer_states[i];
                if (state.frames_seen < static_cast<uint64_t>(clean_exit_after_frames))
                    continue;
            }

            Buffer* buf = buffers[i];

            double last_arrival = buf->get_last_arrival_time();
            // If we are not waiting for data, set last_arrival to start time on first check
            if (!wait_for_first_frame && last_arrival == 0) {
                last_arrival = monitor_start_time;
                DEBUG("monitorBuffer checking buffer {:s}: last arrival time is monitor start time "
                      "{:f}, current time {:f}",
                      buf->buffer_name, last_arrival, cur_time);
            } else {
                DEBUG(
                    "monitorBuffer checking buffer {:s}: last arrival time {:f}, current time {:f}",
                    buf->buffer_name, last_arrival, cur_time);
            }

            if ((cur_time - last_arrival) > timeout && last_arrival > 1) {
                for (auto& buf : buffer_container.get_buffer_map()) {
                    buf.second->print_full_status();
                }
                usleep(50000);
                if (graceful_shutdown) {
                    INFO("The buffer {:s} hasn't received a frame for {:f} seconds.\nPerforming "
                         "graceful shutdown of kotekan because of system timeout.",
                         buf->buffer_name, (cur_time - last_arrival));
                    if (!wait_for_first_frame && last_arrival == monitor_start_time) {
                        WARN("No frames were ever received on buffer {:s}.", buf->buffer_name);
                    }
                    exit_kotekan(CLEAN_EXIT);
                    goto end_loop;
                } else {
                    FATAL_ERROR(
                        "The buffer {:s} hasn't received a frame for {:f} seconds.\nClosing "
                        "kotekan because of system timeout.",
                        buf->buffer_name, (cur_time - last_arrival));
                    goto end_loop;
                }
            }

            uint32_t num_frames = buf->num_frames;
            uint32_t num_full_fames = buf->get_num_full_frames();
            float fraction_full = (float)num_full_fames / (float)num_frames;
            if (fraction_full > fill_threshold) {
                for (auto& buf : buffer_container.get_buffer_map()) {
                    buf.second->print_full_status();
                }
                usleep(50000);
                FATAL_ERROR("The fraction of full frames {:f} ({:d}/{:d}) is greater than the "
                            "threadhold {:f} for buffer: {:s}\nClosing kotekan because of buffer "
                            "fill threadhold exceeded!",
                            fraction_full, num_frames, num_full_fames, fill_threshold,
                            buf->buffer_name);
                goto end_loop;
            }
        }

        // If we are stopping after a number of frames, check that here (for all buffers)
        if (consume_frames && clean_exit_after_frames > 0) {
            bool all_exceeded = true;
            for (const auto& state : buffer_states) {
                if (state.frames_seen < static_cast<uint64_t>(clean_exit_after_frames)) {
                    all_exceeded = false;
                    break;
                }
            }
            if (all_exceeded) {
                INFO("monitorBuffer: All monitored buffers have seen >= {:d} frames, stopping "
                     "monitor.",
                     clean_exit_after_frames);
                exit_kotekan(CLEAN_EXIT);
                goto end_loop;
            }

            // Print out frames counted so far for debugging
            std::string frame_counts = "";
            for (const auto& state : buffer_states) {
                frame_counts += fmt::format("Buffer {:s} has seen {:d} frames. ",
                                            state.buffer->buffer_name, state.frames_seen);
            }
            DEBUG("monitorBuffer: {}", frame_counts);
        }
    }
end_loop:;
}
