#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "nullProcess.hpp"
#include "buffers.h"
#include "errors.h"
#include "output_formating.h"

nullProcess::nullProcess(Config& config, struct Buffer &buf_) :
    KotekanProcess(config, std::bind(&nullProcess::main_thread, this)),
    buf(buf_){
}

nullProcess::~nullProcess() {
}

void nullProcess::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void nullProcess::main_thread() {
    apply_config(0);
    int buffer_ID = 0;

    // Wait for, and drop full buffers
    while (!stop_thread) {

        // This call is blocking!
        buffer_ID = get_full_buffer_from_list(&buf, &buffer_ID, 1);

        // Check if the producer has finished, and we should exit.
        if (buffer_ID == -1) {
            INFO("Closing null thread");
            int ret;
            pthread_exit((void *) &ret);
        }

        INFO("Dropping frame in null thread.");

        mark_buffer_empty(&buf, buffer_ID);

        buffer_ID = (buffer_ID + 1) % buf.num_buffers;
    }
}
