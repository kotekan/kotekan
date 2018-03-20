#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <time.h>
#include <thread>
#include <string>
#include "bufferStatus.hpp"
#include "buffer.h"
#include "errors.h"
#include "visUtil.hpp"
#include "prometheusMetrics.hpp"

REGISTER_KOTEKAN_PROCESS(bufferStatus);

bufferStatus::bufferStatus(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&bufferStatus::main_thread, this)){
    buffers = buffer_container.get_buffer_map();
}

bufferStatus::~bufferStatus() {
}

void bufferStatus::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
    time_delay = config.get_int_default(unique_name, "time_delay", 1000000);
    print_status = config.get_bool_default(unique_name, "print_status", true);
}

void bufferStatus::main_thread() {
    apply_config(0);

    prometheusMetrics &metrics = prometheusMetrics::instance();

    double last_print_time = current_time();

    while (!stop_thread) {
        // Update the metrics every 100ms.
        // Note this puts a lower bound on the status messages.
        usleep(100000);
        double now = current_time();

        for (auto &buf_entry : buffers) {
            uint32_t num_full_frames = get_num_full_frames(buf_entry.second);
            string buffer_name = buf_entry.first;
            metrics.add_process_metric("kotekan_buffer_status_full_frames_total",
                                       unique_name,
                                       num_full_frames,
                                       "buffer_name=\"" + buffer_name + "\"");
            metrics.add_process_metric("kotekan_buffer_status_frames_total",
                                       unique_name,
                                       buf_entry.second->num_frames,
                                       "buffer_name=\"" + buffer_name + "\"");
        }

        if (print_status &&
            (now - last_print_time) > ((double)time_delay / 1000000.0)) {
            last_print_time = now;
            INFO("BUFFER_STATUS");
            for (auto &buf_entry : buffers)
                print_buffer_status(buf_entry.second);
        }
    }
    INFO("Closing Buffer Status thread");
}
