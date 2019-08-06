#include "monitorBuffer.hpp"

#include "util.h"

#include <signal.h>
#include <unistd.h>

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
                ERROR("The buffer %s hasn't received a frame for %f seconds.", buf->buffer_name,
                      (cur_time - last_arrival));
                ERROR("Closing kotekan because of system timeout.");
                for (auto& buf : buffer_container.get_buffer_map()) {
                    print_full_status(buf.second);
                }
                usleep(50000);
                raise(SIGINT);
                goto end_loop;
            }

            uint32_t num_frames = buf->num_frames;
            uint32_t num_full_fames = get_num_full_frames(buf);
            float fraction_full = (float)num_full_fames / (float)num_frames;
            if (fraction_full > fill_threshold) {
                ERROR("The fraction of full frames %f (%d/%d) is greater than the threadhold %f "
                      "for buffer: %s",
                      fraction_full, num_frames, num_full_fames, fill_threshold, buf->buffer_name);
                ERROR("Closing kotekan because of buffer fill threadhold exceeded!");
                for (auto& buf : buffer_container.get_buffer_map()) {
                    print_full_status(buf.second);
                }
                usleep(50000);
                raise(SIGINT);
                goto end_loop;
            }
        }
    }
end_loop:;
}
