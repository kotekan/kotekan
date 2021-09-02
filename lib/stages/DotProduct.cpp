#include "DotProduct.hpp"

#include "StageFactory.hpp"
#include "errors.h"
#include "visUtil.hpp" // for frameID, modulo

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(DotProduct);

STAGE_CONSTRUCTOR(DotProduct) {

    // Register as consumer of in_a_buf and in_b_buf
    in_a_buf = get_buffer("in_a_buf");
    register_consumer(in_a_buf, unique_name.c_str());

    in_b_buf = get_buffer("in_b_buf");
    register_consumer(in_b_buf, unique_name.c_str());

    // Register as a producer of out_buf
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Get the no. of elements in each frame
    _num_elements = in_a_buf->frame_size / sizeof(float);

    // Ensure the input and output buffers are the same
    // length for the dot product
    if(in_a_buf->frame_size != in_b_buf->frame_size) {
        throw std::runtime_error(fmt::format(fmt("in_a_buf frame size does not match in_b_buf frame size. {:d} != {:d}"), in_a_buf->frame_size, in_b_buf->frame_size));
    }

    if(in_a_buf->frame_size != out_buf->frame_size) {
        throw std::runtime_error(fmt::format(fmt("Input frame size does not match output frame size. {:d} != {:d}"), in_a_buf->frame_size, out_buf->frame_size));
    }

    // Load options that can be set in config
    // unique_name_for_stage, name_of_config, default_value_if_not_set
    //_offset = config.get_default<int32_t>(unique_name, "offset", 0);
}

DotProduct::~DotProduct() {}

void DotProduct::main_thread() {

    // Logging function
    INFO("Starting main_thread!");

    // Buffer indices
    frameID in_a_frame_id(in_a_buf);
    frameID in_b_frame_id(in_b_buf);
    frameID out_frame_id(out_buf);

    // Until the thread is stopped
    while (!stop_thread) {

        // Acquire input frames
        uint8_t* frame_a_ptr = wait_for_full_frame(in_a_buf, unique_name.c_str(), in_a_frame_id);
        uint8_t* frame_b_ptr = wait_for_full_frame(in_b_buf, unique_name.c_str(), in_b_frame_id);

        // A null frame is returned on shutdown
        if (frame_a_ptr == nullptr)
            break;
        if (frame_b_ptr == nullptr)
            break;

        // Wait for new output buffer
        uint8_t* out_frame_ptr = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
        if (out_frame_ptr == nullptr)
            break;

        // Cast pointers to float arrays
        float* a = (float*)frame_a_ptr;
        float* b = (float*)frame_b_ptr;
        float* output = (float*)out_frame_ptr;

        // Perform dot product
        for (uint32_t i = 0; i < _num_elements; i++) {
            output[i] = a[i] * b[i];
        }

        // Release the input frames and increment the frame indices
        mark_frame_empty(in_a_buf, unique_name.c_str(), in_a_frame_id++);
        mark_frame_empty(in_b_buf, unique_name.c_str(), in_b_frame_id++);

        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);
    }
}
