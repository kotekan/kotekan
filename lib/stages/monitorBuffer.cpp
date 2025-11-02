#include "monitorBuffer.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer, GenericBuffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for FATAL_ERROR
#include "util.h"              // for e_time

#include "fmt.hpp" // for compile_string_to_view

#include <map>      // for map, operator!=, _Rb_tree_iterator
#include <stdint.h> // for uint32_t
#include <unistd.h> // for usleep, sleep
#include <utility>  // for pair

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

    if (timeout <= 1)
        FATAL_ERROR("monitorBuffer timeout must be > 1 second, got {:d}", timeout);
    if(fill_threshold <= 0.0)
        FATAL_ERROR("monitorBuffer fill_threshold must be > 0.0, got {:f}", fill_threshold);
    if(graceful_shutdown)
        INFO("monitorBuffer configured to perform graceful shutdown on timeout.");
}

monitorBuffer::~monitorBuffer() {}

void monitorBuffer::main_thread() {

    double monitor_start_time = e_time();

    while (!stop_thread) {
        sleep(1);
        double cur_time = e_time();
        for (Buffer* buf : buffers) {
            
            double last_arrival = buf->get_last_arrival_time();
            // If we are not waiting for data, set last_arrival to start time
            if (!wait_for_first_frame && last_arrival == 0) {
                last_arrival = monitor_start_time;
                DEBUG("monitorBuffer checking buffer {:s}: last arrival time is monitor start time {:f}, current time {:f}",
                    buf->buffer_name, last_arrival, cur_time);
            } else {
                DEBUG("monitorBuffer checking buffer {:s}: last arrival time {:f}, current time {:f}",
                    buf->buffer_name, last_arrival, cur_time);
            }
            
            if ((cur_time - last_arrival) > timeout && last_arrival > 1) {
                for (auto& buf : buffer_container.get_buffer_map()) {
                    buf.second->print_full_status();
                }
                usleep(50000);
                if(graceful_shutdown) {
                    INFO("The buffer {:s} hasn't received a frame for {:f} seconds.\nPerforming "
                         "graceful shutdown of kotekan because of system timeout.",
                         buf->buffer_name, (cur_time - last_arrival));
                    if(!wait_for_first_frame && last_arrival == monitor_start_time) {
                        WARN("No frames were ever received on buffer {:s}.", buf->buffer_name);
                    }
                    exit_kotekan(CLEAN_EXIT);
                    goto end_loop;
                }
                FATAL_ERROR("The buffer {:s} hasn't received a frame for {:f} seconds.\nClosing "
                            "kotekan because of system timeout.",
                            buf->buffer_name, (cur_time - last_arrival));
                goto end_loop;
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
    }
end_loop:;
}
