#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "nullProcess.hpp"
#include "buffers.h"
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
    int buffer_ID = 0;

    // Wait for, and drop full buffers
    while (!stop_thread) {

        //INFO("null_process: waiting for buffer");
        buffer_ID = wait_for_full_buffer(buf, unique_name.c_str(), buffer_ID);
        // Check if the producer has finished, and we should exit.
        if (buffer_ID == -1) {
            break;
        }

        //INFO("null_process: Dropping frame %d", buffer_ID);

        //struct ErrorMatrix * error_matrix = get_error_matrix(buf, buffer_ID);
        //INFO("null_process: Dropping frame %d, lost packets: %d", buffer_ID, error_matrix->bad_timesamples);

        //release_info_object(buf, buffer_ID);
        mark_buffer_empty(buf, unique_name.c_str(), buffer_ID);

        buffer_ID = (buffer_ID + 1) % buf->num_buffers;
    }
    INFO("Closing null thread");
}
