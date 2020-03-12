#include "countCheck.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG, FATAL_ERROR
#include "visBuffer.hpp"       // for visFrameView

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdlib.h>   // for llabs
#include <time.h>     // for timespec
#include <tuple>      // for get
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(countCheck);


countCheck::countCheck(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&countCheck::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Fetch tolerance from config.
    start_time_tolerance = config.get_default<int>(unique_name, "start_time_tolerance", 3);

    // Initialize the start_time to zero:
    start_time = 0;
}

void countCheck::main_thread() {

    unsigned int input_frame_id = 0;
    uint64_t counts_per_second = 390625;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        int64_t fpga_seq = std::get<0>(input_frame.time);
        int64_t utime = std::get<1>(input_frame.time).tv_sec;
        int64_t new_start_time = utime - fpga_seq / counts_per_second;

        DEBUG("Debugging: fpga_seq: {:d}, utime: {:d}, start_time: {:d}, time diff: {:d}", fpga_seq,
              utime, start_time, llabs(start_time - new_start_time));

        // If this is the first frame, store start time
        if (start_time == 0) {
            start_time = new_start_time;
            // Else, test that start time is still the same
        } else if (llabs(start_time - new_start_time) > start_time_tolerance) {
            // Shut Kotekan down
            FATAL_ERROR("Found wrong start time. Possible acquisition re-start occurred.");
            break;
        }

        // Mark the buffers and move on
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame ids
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
