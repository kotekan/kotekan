#include "GeoffProducer.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO
#include "visUtil.hpp"        // for frameID, modulo

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t, uint8_t
#include <vector>     // for vector

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(GeoffProducer);

GeoffProducer::GeoffProducer(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GeoffProducer::main_thread, this)) {

    // Register as producer of out_buf
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Load options that can be set in config
    // The arguments to config.get_default are the:
    //     unique_name_for_stage, name_of_config, default_value_if_not_set
    _x_period = config.get_default<float>(unique_name, "x_period", 1.f);
    _speed = config.get_default<float>(unique_name, "speed", 0.f);
    _x0 = config.get_default<float>(unique_name, "x0", 0.f);
    _lo = config.get_default<float>(unique_name, "lo", 0.f);
    _hi = config.get_default<float>(unique_name, "hi", 0.f);
    _width = config.get_default<float>(unique_name, "width", 0.f);
    _type = config.get_default<int>(unique_name, "type", 0);
}


GeoffProducer::~GeoffProducer() {}

// Framework managed pthread
void GeoffProducer::main_thread() {

    // Ring buffer pointer
    frameID frame_id(out_buf);

    // Get the no. of elements in each frame
    uint32_t frame_length = out_buf->frame_size / sizeof(float);

    // Until the thread is stopped
    while (!stop_thread) {

        // Acquire frame
        uint8_t* frame = out_buf->wait_for_empty_frame(unique_name, frame_id);
        // A null frame is returned on shutdown
        if (frame == nullptr)
            break;

        float* data = (float*)frame;

        for (uint32_t i = 0; i < frame_length; i++) {
            float x = (i*_x_period)/frame_length - _speed*frame_id;
            if(_type == 1) {
                //Cosine
                float wavelength = _x_period / ((int) (_x_period/_width));
                data[i] = _lo + 0.5f*(_hi-_lo)*(cosf(2.f*M_PI*(x-_x0) / wavelength) + 1.0f);
            }
            else if(_type == 2) {
                //Square
                float wavelength = _x_period / ((int) (_x_period/_width));
                float y = (x-_x0)/wavelength;
                if(y - floorf(y) < 0.5)
                    data[i] = _hi;
                else
                    data[i] = _lo;
            }
            else if(_type == 3)
            {
                //Gaussian bump
                float y = (x-_x0)/_width;
                data[i] = _lo + (_hi-_lo)*exp(-0.5f*y*y);
            }
            else
                data[i] = _lo;
        }

        INFO("{:s}[{:d}] initialised to: {:f}, ..., {:f}, ..., {:f}", out_buf->buffer_name,
             frame_id, data[0], data[frame_length / 2], data[frame_length - 1]);

        // Release frame
        out_buf->mark_frame_full(unique_name, frame_id);

        // Increase the ring pointer
        frame_id++;
    }
}
