#include "countCheck.hpp"

#include "Config.hpp"          // for Config
#include "N2FrameView.hpp"     // for N2FrameView
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"       // for Telescope
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG, FATAL_ERROR
#include "visBuffer.hpp"       // for VisFrameView

#include "fmt.hpp" // for compile_string_to_view

#include <functional> // for bind, function
#include <stdlib.h>   // for llabs
#include <time.h>     // for timespec
#include <tuple>      // for get


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(countCheck);


countCheck::countCheck(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&countCheck::main_thread, this)) {

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    // Fetch tolerance from config.
    start_time_tolerance = config.get_default<int>(unique_name, "start_time_tolerance", 3);

    // Initialize the start_time to zero:
    start_time_ns = 0;

    tick_len_ns = Telescope::instance().seq_length_nsec();
    if (tick_len_ns == 0) {
        FATAL_ERROR("countCheck: telescope tick length must be > 0.");
    }

    tolerance_ns = (int64_t)start_time_tolerance * 1000000000LL;

    if (in_buf->buffer_type == "vis") {
        input_mode = InputMode::Vis;
    } else if (in_buf->buffer_type == "N2") {
        input_mode = InputMode::N2;
    } else {
        FATAL_ERROR("countCheck input buffer {:s} must be of type vis or N2, got {:s}",
                    in_buf->buffer_name, in_buf->buffer_type);
    }
}

void countCheck::main_thread() {

    unsigned int input_frame_id = 0;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (in_buf->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }

        int64_t new_start_time = 0;

        if (input_mode == InputMode::Vis) {
            // Create view to input frame
            auto input_frame = VisFrameView(in_buf, input_frame_id);

            int64_t fpga_seq = std::get<0>(input_frame.time);
            timespec ts = std::get<1>(input_frame.time);
            new_start_time =
                (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec - fpga_seq * (int64_t)tick_len_ns;

            DEBUG("countCheck (vis): fpga_seq: {:d}, ctime_ns: {:d}, start_time_ns: {:d}, "
                  "time diff: {:d}",
                  fpga_seq, (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec, start_time_ns,
                  llabs(start_time_ns - new_start_time));
        } else {
            // N2 frame view
            auto input_frame = N2FrameView(in_buf, input_frame_id);

            int64_t fpga_seq = (int64_t)input_frame.fpga_start_tick;
            int64_t frame_start_ns = (int64_t)input_frame.frame_start_time_ns;
            new_start_time = frame_start_ns - fpga_seq * (int64_t)tick_len_ns;

            DEBUG("countCheck (N2): fpga_seq: {:d}, frame_start_ns: {:d}, start_time_ns: {:d}, "
                  "time diff: {:d}",
                  fpga_seq, frame_start_ns, start_time_ns, llabs(start_time_ns - new_start_time));
        }

        // If this is the first frame, store start time
        if (start_time_ns == 0) {
            start_time_ns = new_start_time;
            // Else, test that start time is still the same
        } else if (llabs(start_time_ns - new_start_time) > tolerance_ns) {
            // Shut Kotekan down
            FATAL_ERROR("Found wrong start time. Possible acquisition re-start occurred.");
            break;
        }

        // Mark the buffers and move on
        in_buf->mark_frame_empty(unique_name, input_frame_id);

        // Advance the current frame ids
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}
