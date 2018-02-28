#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <time.h>
#include <thread>
#include "bufferStatus.hpp"
#include "buffer.h"
#include "errors.h"
#include "output_formating.h"

REGISTER_KOTEKAN_PROCESS(bufferStatus);

PROCESS_CONSTRUCTOR(bufferStatus) {
    buffers = buffer_container.get_buffer_map();
    time_delay = config.get_int_default(unique_name,"time_delay",100000);
}

bufferStatus::~bufferStatus() {
}

void bufferStatus::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
}

void bufferStatus::main_thread() {
    apply_config(0);

    // Wait for, and drop full buffers
    while (!stop_thread) {
        usleep(time_delay);
        DEBUG("BUFFER_STATUS");
        for (auto &buf : buffers)
        {
            print_buffer_status(buf.second);
        }
    }
    DEBUG("Closing Buffer Status thread");
}
