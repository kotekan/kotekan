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
using kotekan::prometheusMetrics;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(visDebug);


visDebug::visDebug(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visDebug::main_thread, this)) {

    // Setup the input vector
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

void visDebug::main_thread() {

    unsigned int frame_id = 0;

    uint64_t num_frames = 0;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
        if ((num_frames % 1000) == 0)
            INFO("Got frame number %lli", num_frames);
        auto frame = visFrameView(in_buf, frame_id);
        DEBUG("%s", frame.summary().c_str());

        // Update the frame count for prometheus
        fd_pair key{frame.freq_id, frame.dataset_id};
        frame_counts[key]++; // Relies on the fact that insertion zero intialises
        std::string labels =
            fmt::format("freq_id=\"{}\",dataset_id=\"{}\"", frame.freq_id, frame.dataset_id);
        prometheusMetrics::instance().add_stage_metric("kotekan_visdebug_frame_total", unique_name,
                                                       frame_counts[key], labels);

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        frame_id = (frame_id + 1) % in_buf->num_frames;
        num_frames++;
    }
}
