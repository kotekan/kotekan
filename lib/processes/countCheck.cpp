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

    // Initialize the start_time to zero:
    start_time = 0;

    // Fetch any simple configuration
    test_nframes = config.get_int_default(unique_name, "test_nframes", -1);

}

void countCheck::apply_config(uint64_t fpga_seq) {

}

void countCheck::main_thread() {

    unsigned int input_frame_id = 0;
    uint64_t counts_per_second = 390625;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        int64_t fpga_seq = std::get<0>(input_frame.time);
        int64_t utime = std::get<1>(input_frame.time).tv_sec;
        int64_t new_start_time = utime - fpga_seq/counts_per_second;

        // If this is the first frame, store start time
        if (start_time == 0) {
            start_time = new_start_time;
        // Else, test that start time is still the same
        } else if ( llabs(start_time - new_start_time) > 3 ) {
            INFO("Found wrong start time. Possible acquisition re-start occurred.");
            INFO("Stopping Kotekan.");
            // Shut Kotekan down
            std::raise(SIGINT);
        }

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;

        // In case it's a test run
        if((test_nframes != -1) && (test_nframes==input_frame_id)) {
            INFO("This is a counteCheck test run. Shutting down Kotekan!");
            // Shut Kotekan down
            std::raise(SIGINT);
        }

        

    }

}