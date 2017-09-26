#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "nullProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "output_formating.h"

nullProcess::nullProcess(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&nullProcess::main_thread, this)){
    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
}

nullProcess::~nullProcess() {
}

void nullProcess::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void nullProcess::main_thread() {
    apply_config(0);
    int frame_id = 0;

    // Wait for, and drop full buffers
    while (!stop_thread) {

        INFO("null_process: waiting for buffer");
        wait_for_full_frame(buf, unique_name.c_str(), frame_id);

        INFO("null_process: Dropping frame %s[%d]", buf->buffer_name, frame_id);

        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + 1) % buf->num_frames;
    }
    INFO("Closing null thread");
}
