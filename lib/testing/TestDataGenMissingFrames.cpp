#include "TestDataGenMissingFrames.hpp"

#include "chimeMetadata.h"
#include "errors.h"
#include "fpga_header_functions.h"

#include <math.h>
#include <stdlib.h>
#include <unistd.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(TestDataGenMissingFrames);

TestDataGenMissingFrames::TestDataGenMissingFrames(Config& config, const string& unique_name,
                                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&TestDataGenMissingFrames::main_thread, this)),
    _samples_per_data_set(config.get<uint32_t>(unique_name, "samples_per_data_set")),
    _missing_frames(config.get_default<std::vector<uint32_t>>(unique_name, "missing_frames", {})) {
    INFO("Stage will drop {:d} frames.", _missing_frames.size());

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void TestDataGenMissingFrames::main_thread() {
    int frame_count = 0;
    int in_buf_id = 0;
    int out_buf_id = 0;

    while (!stop_thread) {
        uint8_t* input = wait_for_full_frame(in_buf, unique_name.c_str(), in_buf_id);
        if (input == NULL)
            break;

        // Copy the frame, unless it's in the list of frames to drop
        if (std::find(_missing_frames.begin(), _missing_frames.end(), frame_count)
            == _missing_frames.end()) {
            float* output = (float*)wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_id);
            if (output == NULL)
                break;

            memcpy(output, input, out_buf->frame_size);

            pass_metadata(in_buf, in_buf_id, out_buf, out_buf_id);

            mark_frame_full(out_buf, unique_name.c_str(), out_buf_id);
            out_buf_id = (out_buf_id + 1) % out_buf->num_frames;
        } else {
            INFO("Add missing frame {}!!!", frame_count);
        }

        mark_frame_empty(in_buf, unique_name.c_str(), in_buf_id);
        frame_count++;
        in_buf_id = frame_count % in_buf->num_frames;

    } // end stop thread
}
