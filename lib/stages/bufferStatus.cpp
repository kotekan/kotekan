#include "bufferStatus.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for get_num_full_frames, print_buffer_status, Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for INFO
#include "prometheusMetrics.hpp" // for Metrics, Gauge, MetricFamily
#include "visUtil.hpp"           // for current_time

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t
#include <string>     // for string, allocator
#include <unistd.h>   // for usleep
#include <utility>    // for pair
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(bufferStatus);

bufferStatus::bufferStatus(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferStatus::main_thread, this)) {
    buffers = buffer_container.get_basic_buffer_map();

    // Apply config.
    time_delay = config.get_default<int>(unique_name, "time_delay", 1000000);
    print_status = config.get_default<bool>(unique_name, "print_status", true);
}

bufferStatus::~bufferStatus() {}

void bufferStatus::main_thread() {

    Metrics& metrics = Metrics::instance();
    auto& frames_counter =
        metrics.add_gauge("kotekan_bufferstatus_frames_total", unique_name, {"buffer_name"});
    auto& full_frames_counter =
        metrics.add_gauge("kotekan_bufferstatus_full_frames_total", unique_name, {"buffer_name"});

    double last_print_time = current_time();

    while (!stop_thread) {
        // Update the metrics every 100ms.
        // Note this puts a lower bound on the status messages.
        usleep(100000);
        double now = current_time();

        for (auto& buf_entry : buffers) {
            uint32_t num_full_frames = buf_entry.second->get_num_full_frames();
            std::string buffer_name = buf_entry.first;
            full_frames_counter.labels({buffer_name}).set(num_full_frames);
            frames_counter.labels({buffer_name}).set(buf_entry.second->num_frames);
        }

        if (print_status && (now - last_print_time) > ((double)time_delay / 1000000.0)) {
            last_print_time = now;
            INFO("BUFFER_STATUS");
            for (auto& buf_entry : buffers)
                buf_entry.second->print_buffer_status();
        }
    }
    INFO("Closing Buffer Status thread");
}
