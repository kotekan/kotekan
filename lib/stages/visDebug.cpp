#include "visDebug.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t
#include "kotekanLogging.hpp"    // for DEBUG, INFO
#include "prometheusMetrics.hpp" // for Metrics, Counter, MetricFamily
#include "visBuffer.hpp"         // for VisFrameView

#include <atomic>     // for atomic_bool
#include <cstdint>    // for uint64_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(visDebug);


visDebug::visDebug(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visDebug::main_thread, this)) {

    // Setup the input vector
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    _output_period = config.get_default<int>(unique_name, "output_period", 1000);
}

void visDebug::main_thread() {

    unsigned int frame_id = 0;

    uint64_t num_frames = 0;

    auto& frame_freq_counter = Metrics::instance().add_counter(
        "kotekan_visdebug_frames_by_freq_total", unique_name, {"freq_id"});

    auto& frame_dataset_counter = Metrics::instance().add_counter(
        "kotekan_visdebug_frames_by_dataset_total", unique_name, {"dataset_id"});
    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Print out debug information from the buffer
        if ((num_frames % _output_period) == 0)
            INFO("Got frame number {:d}", num_frames);
        auto frame = VisFrameView(in_buf, frame_id);
        DEBUG("{:s}", frame.summary());

        frame_freq_counter.labels({std::to_string(frame.freq_id)}).inc();
        frame_dataset_counter.labels({frame.dataset_id.to_string()}).inc();

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        frame_id = (frame_id + 1) % in_buf->num_frames;
        num_frames++;
    }
}
