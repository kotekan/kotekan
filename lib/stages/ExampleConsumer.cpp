#include "ExampleConsumer.hpp"

#include "StageFactory.hpp"
#include "errors.h"
#include "visUtil.hpp" // for frameID, modulo

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(ExampleConsumer);

ExampleConsumer::ExampleConsumer(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ExampleConsumer::main_thread, this)) {

    // Register as consumer of in_buf
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Get the no. of elements in each frame
    _num_elements = in_buf->frame_size / sizeof(float);
}


ExampleConsumer::~ExampleConsumer() {}

// Framework managed pthread
void ExampleConsumer::main_thread() {
    // Logging function
    INFO("Reached main_thread!");

    // Ring buffer pointer
    frameID frame_id(in_buf);

    // Until the thread is stopped
    while (!stop_thread) {

        // Acquire frame
        uint8_t* frame = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        // A null frame is returned on shutdown
        if (frame == nullptr)
            break;

        float* data = (float*)frame;

        // Logging
        INFO("{:s}[{:d}]: {:f}, ..., {:f}, ..., {:f}", in_buf->buffer_name, frame_id, data[0],
             data[_num_elements / 2], data[_num_elements - 1]);

        // Release frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Increase the ring pointer
        frame_id++;
    }
}
