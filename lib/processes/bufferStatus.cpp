#include "bufferStatus.hpp"

#include "buffer.h"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp"

#include <fcntl.h>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::prometheusMetrics;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferStatus);

bufferStatus::bufferStatus(Config& config, const string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferStatus::main_thread, this)) {
    buffers = buffer_container.get_buffer_map();
}

bufferStatus::~bufferStatus() {}

void bufferStatus::main_thread() {

    // Apply config.
    time_delay = config.get_default<int>(unique_name, "time_delay", 1000000);
    print_status = config.get_default<bool>(unique_name, "print_status", true);

    prometheusMetrics& metrics = prometheusMetrics::instance();

    double last_print_time = current_time();

    while (!stop_thread) {
        // Update the metrics every 100ms.
        // Note this puts a lower bound on the status messages.
        usleep(100000);
        double now = current_time();

        for (auto& buf_entry : buffers) {
            uint32_t num_full_frames = get_num_full_frames(buf_entry.second);
            string buffer_name = buf_entry.first;
            metrics.add_stage_metric("kotekan_bufferstatus_full_frames_total", unique_name,
                                     num_full_frames, "buffer_name=\"" + buffer_name + "\"");
            metrics.add_stage_metric("kotekan_bufferstatus_frames_total", unique_name,
                                     buf_entry.second->num_frames,
                                     "buffer_name=\"" + buffer_name + "\"");
        }

        if (print_status && (now - last_print_time) > ((double)time_delay / 1000000.0)) {
            last_print_time = now;
            INFO("BUFFER_STATUS");
            for (auto& buf_entry : buffers)
                print_buffer_status(buf_entry.second);
        }
    }
    INFO("Closing Buffer Status thread");
}
