#include "TestDropFrames.hpp"

#include "Config.hpp"         // for Config
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"           // for Buffer, get_num_consumers, mark_frame_empty, mark_frame_full
#include "kotekanLogging.hpp" // for INFO

#include <algorithm>     // for find
#include <atomic>        // for atomic_bool
#include <cstdint>       // for uint32_t
#include <cstring>       // for memcpy
#include <exception>     // for exception
#include <functional>    // for _Bind_helper<>::type, bind, function
#include <random>        // for default_random_engine, bernoulli_distribution, random_device
#include <regex>         // for match_results<>::_Base_type
#include <stdexcept>     // for runtime_error
#include <unordered_set> // for unordered_set


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(TestDropFrames);

TestDropFrames::TestDropFrames(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&TestDropFrames::main_thread, this)),
    _samples_per_data_set(config.get<uint32_t>(unique_name, "samples_per_data_set")),
    _missing_frames(config.get_default<std::vector<uint32_t>>(unique_name, "missing_frames", {})),
    _drop_frame_chance(config.get_default<double>(unique_name, "drop_frame_chance", 0)) {
    INFO("Stage will drop {:d} / {:.2}% frames.", _missing_frames.size(), _drop_frame_chance * 100);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void TestDropFrames::main_thread() {
    int frame_count = 0;
    int in_buf_id = 0;
    int out_buf_id = 0;

    std::random_device rd;
    std::default_random_engine gen(rd());
    std::bernoulli_distribution draw_frame_drop(_drop_frame_chance);

    // use a set type for quicker lookup of frames to drop
    const auto missing_frames =
        std::unordered_set<uint32_t>(_missing_frames.cbegin(), _missing_frames.cend());

    while (!stop_thread) {
        uint8_t* input = wait_for_full_frame(in_buf, unique_name.c_str(), in_buf_id);
        if (input == nullptr)
            break;

        // Copy the frame, unless it's in the list of frames to drop or it drew the "DROP" odds
        if (missing_frames.size() && missing_frames.count(frame_count)) {
            INFO("Drop frame {} because it is in the missing_frame list.", frame_count);
        } else if (_drop_frame_chance && draw_frame_drop(gen)) {
            INFO("Drop frame {} because it drew the short straw.", frame_count);
        } else {
            uint8_t* output = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_id);
            if (output == nullptr)
                break;

            int num_consumers = get_num_consumers(in_buf);

            // Copy or transfer the data part.
            if (num_consumers == 1) {
                // Transfer frame contents with directly...
                swap_frames(in_buf, in_buf_id, out_buf, out_buf_id);
            } else if (num_consumers > 1) {
                // Copy the frame data over, leaving the source intact
                std::memcpy(output, input, in_buf->frame_size);
            }

            pass_metadata(in_buf, in_buf_id, out_buf, out_buf_id);

            mark_frame_full(out_buf, unique_name.c_str(), out_buf_id);
            out_buf_id = (out_buf_id + 1) % out_buf->num_frames;
        }

        mark_frame_empty(in_buf, unique_name.c_str(), in_buf_id);
        frame_count++;
        in_buf_id = frame_count % in_buf->num_frames;

    } // end stop thread
}
