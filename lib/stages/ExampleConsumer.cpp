#include "ExampleConsumer.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO
#include "visUtil.hpp"        // for frameID, modulo

#include <atomic>     // for atomic_bool
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdint.h>   // for uint32_t, uint8_t

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(ExampleConsumer);

/*
 * ExampleConsumer constructor.  Note that you can instead use the macro
 *
 *    STAGE_CONSTRUCTOR(ExampleConsumer)
 *
 * which saves the boilerplate of the constructor signature.
 */
ExampleConsumer::ExampleConsumer(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ExampleConsumer::main_thread, this)) {

    // Register as consumer of in_buf
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
}


ExampleConsumer::~ExampleConsumer() {}

// Framework managed pthread
void ExampleConsumer::main_thread() {
    // Logging function
    INFO("Reached main_thread!");

    // Ring buffer pointer
    frameID frame_id(in_buf);

    // Get the no. of elements in each frame
    uint32_t frame_length = in_buf->frame_size / sizeof(float);

    // Until the thread is stopped
    while (!stop_thread) {

        // Acquire frame
        uint8_t* frame = in_buf->wait_for_full_frame(unique_name, frame_id);
        // A null frame is returned on shutdown
        if (frame == nullptr)
            break;

        float* data = (float*)frame;

        // Logging
        INFO("{:s}[{:d}]: {:f}, ..., {:f}, ..., {:f}", in_buf->buffer_name, frame_id, data[0],
             data[frame_length / 2], data[frame_length - 1]);

        // Release frame
        in_buf->mark_frame_empty(unique_name, frame_id);

        // Increase the ring pointer
        frame_id++;
    }
}
