#include "TimeUtilDump.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO
#include "timeUtil.hpp"
#include "visUtil.hpp" // for frameID, modulo

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(TimeUtilDump);

TimeUtilDump::TimeUtilDump(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&TimeUtilDump::main_thread, this)) {

    // Register as consumer of in_buf
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    // Register as a producer of out_buf
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Ensure the output buffer length matches the input buffer lengths
    if (3 * in_buf->frame_size != out_buf->frame_size) {
        throw std::runtime_error(
            fmt::format(fmt("Input frame size does not match output frame size. 2x{:d} != {:d}"),
                        in_buf->frame_size, out_buf->frame_size));
    }

    _dUT = config.get_default<double>(unique_name, "dUT_sec", 0.0);
    _dAT = config.get_default<double>(unique_name, "dAT_sec", 0.0);
}


TimeUtilDump::~TimeUtilDump() {}

// Framework managed pthread
void TimeUtilDump::main_thread() {
    // Logging function
    INFO("Reached main_thread!");

    // Buffer indices
    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    // Length of vectors
    uint32_t frame_length = in_buf->frame_size / (2 * sizeof(long));

    // Until the thread is stopped
    while (!stop_thread) {

        // Acquire input frames
        uint8_t* frame_ptr = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        // A null frame is returned on shutdown
        if (frame_ptr == nullptr)
            break;

        // Wait for new output buffer
        uint8_t* out_frame_ptr = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame_ptr == nullptr)
            break;

        long* input = (long*)frame_ptr;
        long* output = (long*)out_frame_ptr;

        for (uint32_t i = 0; i < frame_length; i++) {
            timespec time{.tv_sec = input[2 * i], .tv_nsec = input[2 * i + 1]};
            INFO("Instrument Time: {:d} s {:d} ns", time.tv_sec, time.tv_nsec);
            timespec ut1 = get_UT1_from_time(time, _dUT - _dAT);
            double era = get_ERA_from_UT1(ut1, nullptr);
            double era2 = get_ERA_from_time(time, _dUT - _dAT);
            output[6 * i + 0] = time.tv_sec;
            output[6 * i + 1] = time.tv_nsec;
            output[6 * i + 2] = ut1.tv_sec;
            output[6 * i + 3] = ut1.tv_nsec;
            memcpy(&(output[6 * i + 4]), &era, sizeof(long));
            memcpy(&(output[6 * i + 5]), &era2, sizeof(long));
        }

        // Release the input frames and increment the frame indices
        in_buf->mark_frame_empty(unique_name, in_frame_id++);

        // Release the output frame and increment the output frame index
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }
}
