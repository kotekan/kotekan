#include "monitorBuffer.hpp"
#include "util.h"

#include <unistd.h>
#include <signal.h>

REGISTER_KOTEKAN_PROCESS(monitorBuffer);

PROCESS_CONSTRUCTOR(monitorBuffer) {

    // Note we do not register as a producer or consumer here.
    buffers = get_buffer_array("bufs");
    // Timeout is in seconds
    timeout = config.get_int_default(unique_name, "timeout", 60);

}

monitorBuffer::~monitorBuffer() {
}

void monitorBuffer::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void monitorBuffer::main_thread() {
    apply_config(0);

    // Wait for, and drop full buffers
    while (!stop_thread) {
        sleep(1);
        double cur_time = e_time();
        for (struct Buffer * buf : buffers) {
            double last_arrival = get_last_arrival_time(buf);
            if ((cur_time - last_arrival) > timeout && last_arrival > 1) {
                ERROR("The buffer %s hasn't received a frame for %f seconds.",
                      buf->buffer_name, (cur_time - last_arrival));
                ERROR("Closing kotekan because of system timeout.");
                for (auto &buf : buffer_container.get_buffer_map()) {
                    print_buffer_status(buf.second);
                }
                usleep(50000);
                raise(SIGINT);
                goto end_loop;
            }
        }
    }
    end_loop:;
}
