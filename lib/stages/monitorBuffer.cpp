#include "monitorBuffer.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for print_full_status, Buffer, get_last_arrival_time, get_num...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for FATAL_ERROR
#include "util.h"              // for e_time

#include <atomic>    // for atomic_bool
#include <exception> // for exception
#include <map>       // for map
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t
#include <unistd.h>  // for usleep, sleep
#include <utility>   // for pair

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_KOTEKAN_STAGE(monitorBuffer);

STAGE_CONSTRUCTOR(monitorBuffer) {

    // Note we do not register as a producer or consumer here.
    buffers = get_buffer_array("bufs");
    // Timeout is in seconds
    timeout = config.get_default<int>(unique_name, "timeout", 60);
    fill_threshold = config.get_default<float>(unique_name, "fill_threshold", 2.0);
}

monitorBuffer::~monitorBuffer() {}

void monitorBuffer::main_thread() {

    while (!stop_thread) {
        sleep(1);
        double cur_time = e_time();
        for (struct Buffer* buf : buffers) {
            double last_arrival = get_last_arrival_time(buf);
            if ((cur_time - last_arrival) > timeout && last_arrival > 1) {
                for (auto& buf : buffer_container.get_buffer_map()) {
                    print_full_status(buf.second);
                }
                usleep(50000);
                FATAL_ERROR("The buffer {:s} hasn't received a frame for {:f} seconds.\nClosing "
                            "kotekan because of system timeout.",
                            buf->buffer_name, (cur_time - last_arrival));
                goto end_loop;
            }

            uint32_t num_frames = buf->num_frames;
            uint32_t num_full_fames = get_num_full_frames(buf);
            float fraction_full = (float)num_full_fames / (float)num_frames;
            if (fraction_full > fill_threshold) {
                for (auto& buf : buffer_container.get_buffer_map()) {
                    print_full_status(buf.second);
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
