#include "ExampleProducer.hpp"
#include <stdint.h>            // for uint32_t, uint8_t
#include <atomic>              // for atomic_bool
#include <exception>           // for exception
#include <functional>          // for _Bind_helper<>::type, bind, function
#include <regex>               // for match_results<>::_Base_type
#include <stdexcept>           // for runtime_error
#include <vector>              // for vector
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp"  // for INFO
#include "visUtil.hpp"         // for frameID, modulo

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(ExampleProducer);

ExampleProducer::ExampleProducer(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ExampleProducer::main_thread, this)) {

    // Register as producer of out_buf
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Load options that can be set in config
    // The arguments to config.get_default are the:
    //     unique_name_for_stage, name_of_config, default_value_if_not_set
    _init_value = config.get_default<float>(unique_name, "init_value", 0.f);
}


ExampleProducer::~ExampleProducer() {}

// Framework managed pthread
void ExampleProducer::main_thread() {

    // Ring buffer pointer
    frameID frame_id(out_buf);

    // Get the no. of elements in each frame
    uint32_t frame_length = out_buf->frame_size / sizeof(float);

    // Until the thread is stopped
    while (!stop_thread) {

        // Acquire frame
        uint8_t* frame = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id);
        // A null frame is returned on shutdown
        if (frame == nullptr)
            break;

        float* data = (float*)frame;

        for (uint32_t i = 0; i < frame_length; i++) {
            data[i] = _init_value;
        }

        INFO("{:s}[{:d}] initialised to: {:f}, ..., {:f}, ..., {:f}", out_buf->buffer_name,
             frame_id, data[0], data[frame_length / 2], data[frame_length - 1]);

        // Release frame
        mark_frame_full(out_buf, unique_name.c_str(), frame_id);

        // Increase the ring pointer
        frame_id++;
    }
}
