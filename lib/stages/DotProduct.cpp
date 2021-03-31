#include "DotProduct.hpp"

#include "StageFactory.hpp"
#include "errors.h"
#include "visUtil.hpp" // for frameID, modulo

/* Load the classes we will be using
 * TODO: Annotate the includes and usings */

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with abstract factory.
// Needed for Kotekan to be able to find the stage.
REGISTER_KOTEKAN_STAGE(DotProduct);

/* Constructor for the stage
    TODO: Annotate the arguments everywhere
    Note, that you can use the macro STAGE_CONSTRUCTOR(DotProduct)
    if your constructor does not need additional customisation
    and you wish to hide the complexity. */
DotProduct::DotProduct(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&DotProduct::main_thread, this)) {

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
    assert(in_a_buf->frame_size == in_b_buf->frame_size
           && in_a_buf->frame_size == out_buf->frame_size);

    // Load options that can be set in config
    // unique_name_for_stage, name_of_config, default_value_if_not_set
    //_offset = config.get_default<int32_t>(unique_name, "offset", 0);
}

// Deconstructor; what happens when Kotekan shuts down
DotProduct::~DotProduct() {}

// Framework managed pthread
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

        // Get pointers to float values
        float* a = (float*)frame_a_ptr;
        float* b = (float*)frame_b_ptr;
        float* output = (float*)out_frame_ptr;

        // Perform dot product
        for (uint32_t i = 0; i < _num_elements; i++) {
            output[i] = a[i] * b[i];
        }

        // Logging
        DEBUG("Input buffer {:s}[{:d}]", in_a_buf->buffer_name, in_a_frame_id);
        DEBUG("Input buffer {:s}[{:d}]", in_b_buf->buffer_name, in_b_frame_id);
        INFO("Output buffer {:s}[{:d}]: {:f}, ..., {:f}, ..., {:f}", out_buf->buffer_name,
             out_frame_id, output[0], output[_num_elements / 2], output[_num_elements - 1]);

        // Release the input frames and increment the frame indices
        mark_frame_empty(in_a_buf, unique_name.c_str(), in_a_frame_id++);
        mark_frame_empty(in_b_buf, unique_name.c_str(), in_b_frame_id++);

        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);
    }
}
