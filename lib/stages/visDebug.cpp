#include "visDebug.hpp"

#include "StageFactory.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "visBuffer.hpp"

#include "fmt.hpp"

#include <atomic>
#include <functional>
#include <stdint.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(visDebug);


visDebug::visDebug(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visDebug::main_thread, this)) {

    // Setup the input vector
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    _output_period = config.get_default<int>(unique_name, "output_period", 1000);
}

void visDebug::main_thread() {

    unsigned int frame_id = 0;

    uint64_t num_frames = 0;

    auto& frame_counter = Metrics::instance().add_counter("kotekan_visdebug_frame_total",
                                                          unique_name, {"freq_id", "dataset_id"});
    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
        if ((num_frames % _output_period) == 0)
            INFO("Got frame number {:d}", num_frames);
        auto frame = visFrameView(in_buf, frame_id);
        DEBUG("{:s}", frame.summary());

        frame_counter.labels({std::to_string(frame.freq_id), frame.dataset_id.to_string()}).inc();

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        frame_id = (frame_id + 1) % in_buf->num_frames;
        num_frames++;
    }
}
