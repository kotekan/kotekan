#include "bkCheck.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG, INFO

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdlib.h>   // for free, malloc
#include <vector>     // for vector
#include <iostream>


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bkCheck);

bkCheck::bkCheck(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bkCheck::main_thread, this)) {

    // Apply config.
    // _num_elements = config.get<int32_t>(unique_name, "num_elements");
    // _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    // _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    // _num_blocks = config.get<int32_t>(unique_name, "num_blocks");
    // _block_size = config.get<int32_t>(unique_name, "block_size");
    // _data_format = config.get_default<std::string>(unique_name, "data_format", "4+4b");

    _input_buf = get_buffer("network_in_buf");
    register_consumer(_input_buf, unique_name.c_str());
}

bkCheck::~bkCheck() {
}

void bkCheck::main_thread() {
    std::cout << _input_buf << "\n";
    
    int input_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        float* input = (float*)wait_for_full_frame(_input_buf, unique_name.c_str(), input_frame_id);
        if (input == nullptr)
            break;
        // int* output = (int*)wait_for_empty_frame(output_buf, unique_name.c_str(), output_frame_id);
        // if (output == nullptr)
        //     break;


        std::cout << "bkCheck: " << input[0] << "\n";



        // TODO adjust to allow for more than one frequency.
        // TODO remove all the 32's in here with some kind of constant/define
        // INFO("Simulating GPU processing for {:s}[{:d}] putting result in {:s}[{:d}]",
        //      input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        // INFO("Simulating GPU processing done for {:s}[{:d}] result is in {:s}[{:d}]",
        //      input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        // pass_metadata(input_buf, input_frame_id, output_buf, output_frame_id);
        mark_frame_empty(_input_buf, unique_name.c_str(), input_frame_id);
        // mark_frame_full(output_buf, unique_name.c_str(), output_frame_id);

        input_frame_id = (input_frame_id + 1) % _input_buf->num_frames;
        // output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
