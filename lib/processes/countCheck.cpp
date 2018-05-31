#include "countCheck.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "chimeMetadata.h"
#include <algorithm>
#include <csignal>

REGISTER_KOTEKAN_PROCESS(countCheck);


countCheck::countCheck(Config& config,
                     const string& unique_name,
                     bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&countCheck::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Initiate the previous FPGA seuqnce count to zero:
    prev_fpga_seq = 0;

}

void countCheck::apply_config(uint64_t fpga_seq) {

}

void countCheck::main_thread() {

    unsigned int input_frame_id = 0;
    uint64_t counts_per_second = 390625;
    //uint64_t counts_per_second = 3;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        uint64_t fpga_seq = get_fpga_seq_num(in_buf, input_frame_id);

//        INFO("Prev seq num = %i, current seq num = %i, tolerance = %i", 
//                prev_fpga_seq, 
//                fpga_seq, 
//                (int64_t)(fpga_seq-(counts_per_second*3600)) );
       
        // Need to convert to signed int to prevent errors when value is negative
        if((int64_t)prev_fpga_seq < (int64_t)(fpga_seq-(counts_per_second*3600))) {
            INFO("Current frame has FPGA count more than 1 hour behind previous one. Stopping Kotekan.");
            // Shut Kotekan down:
            std::raise(SIGINT);
        }

        // Update value for the previous FPGA sequence count: 
        prev_fpga_seq = fpga_seq;

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }

}


