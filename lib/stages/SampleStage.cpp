#include "SampleStage.hpp"
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
REGISTER_KOTEKAN_STAGE(SampleStage);

/* Constructor for the stage
    TODO: Annotate the arguments everywhere
    Note, that you can use the macro STAGE_CONSTRUCTOR(SampleStage)
    if your constructor does not need additional customisation
    and you wish to hide the complexity */
SampleStage::SampleStage(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, 
            std::bind(&SampleStage::main_thread, this)) {

        // Register as consumer of in_buf
        in_buf = get_buffer("in_buf");
        register_consumer(in_buf, unique_name.c_str());

        // Load options that can be set in config
        // unique_name_for_stage, name_of_config, default_value_if_not_set
        _len = config.get_default<int32_t>(unique_name, "len", 128);
        _offset = config.get_default<int32_t>(unique_name, "offset", 0);
    }

// Deconstructor; what happens when Kotekan shuts down
SampleStage::~SampleStage() {}

// Framework managed pthread
void SampleStage::main_thread() {
    // Logging function
    INFO("Sample Stage, reached main_thread!");

    // Ring buffer pointer
    frameID frame_id(in_buf);

    // Until the thread is stopped
    while (!stop_thread) {
        INFO("In thread!");

        // Acquire frame
        uint8_t* frame = wait_for_full_frame(in_buf,
                                            unique_name.c_str(),
                                            frame_id);
        // A null frame is returned on shutdown
        if (frame == NULL)
            break;

        // Logging
        DEBUG("SampleStage: Got buffer {:s}[{:d}]", in_buf->buffer_name, frame_id);

        // Release frame
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Increase the ring pointer
        frame_id++;
    }
}
