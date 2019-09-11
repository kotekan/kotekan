#include "testDataGenMissingFrames.hpp"

#include "chimeMetadata.h"
#include "errors.h"
#include "fpga_header_functions.h"

#include <math.h>
#include <stdlib.h>
#include <unistd.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataGenMissingFrames);

testDataGenMissingFrames::testDataGenMissingFrames(Config& config, const string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&testDataGenMissingFrames::main_thread, this)) {

    // Apply config.
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _missing_frame_index = config.get<uint32_t>(unique_name, "missing_frame_index");

    input_buf = get_buffer("input_buf");
    register_consumer(input_buf, unique_name.c_str());
    output_buf = get_buffer("output_buf");
    register_producer(output_buf, unique_name.c_str());
    
}

testDataGenMissingFrames::~testDataGenMissingFrames() {
}

void testDataGenMissingFrames::main_thread() {
    int input_buf_id = 0;
    int output_buf_id = 0;
    int num_frames = 0;

    while (!stop_thread) {
        float* input =
            (float*)wait_for_full_frame(input_buf, unique_name.c_str(), input_buf_id);
        if (input == NULL)
            break;

        if((num_frames + 1) % _missing_frame_index != 0) {
          float* output =
            (float*)wait_for_empty_frame(output_buf, unique_name.c_str(), output_buf_id);
          if (output == NULL)
            break;

          memcpy(output, input, output_buf->frame_size);

          pass_metadata(input_buf, input_buf_id, output_buf, output_buf_id);

          mark_frame_full(output_buf, unique_name.c_str(), output_buf_id);
          output_buf_id = (output_buf_id + 1) % output_buf->num_frames;
        }
        else {
          INFO("\nAdded missing frame!!!\n");
        }

        num_frames++;
        mark_frame_empty(input_buf, unique_name.c_str(), input_buf_id);
        input_buf_id = (input_buf_id + 1) % input_buf->num_frames;

    } // end stop thread
}
