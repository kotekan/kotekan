#include "visDrop.hpp"

#include "StageFactory.hpp"
#include "errors.h"
#include "visBuffer.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(visDrop);

visDrop::visDrop(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visDrop::main_thread, this)) {

    // Setup the buffers
    buf_in = get_buffer("in_buf");
    register_consumer(buf_in, unique_name.c_str());
    buf_out = get_buffer("out_buf");
    register_producer(buf_out, unique_name.c_str());

    drop_freqs = config.get_default<std::vector<uint32_t>>(unique_name, "freq", {});
    INFO("Dropping {:d} frequencies.", drop_freqs.size());
    frac_rfi = config.get_default<float>(unique_name, "frac_rfi", 0.);
    frac_lost = config.get_default<float>(unique_name, "frac_lost", 0.);
}

void visDrop::main_thread() {

    uint32_t frame_id_in = 0;
    uint32_t frame_id_out = 0;

    while (!stop_thread) {
        // Wait for data in the input buffer
        if ((wait_for_full_frame(buf_in, unique_name.c_str(), frame_id_in)) == nullptr) {
            break;
        }

        // Wait for space in the output buffer
        if (wait_for_empty_frame(buf_out, unique_name.c_str(), frame_id_out) == nullptr) {
            break;
        }
        // Copy frame into output buffer
        auto frame = visFrameView::copy_frame(buf_in, frame_id_in, buf_out, frame_id_out);

        // Check if this frame should be dropped because of its freq_id.
        if (std::find(drop_freqs.begin(), drop_freqs.end(), frame.freq_id) != drop_freqs.end()) {
            if (frac_lost != 0.) {
                DEBUG("Setting lost samples for frame {:d} with frequency ID {:d}.", frame_id_in,
                      frame.freq_id);
                frame.fpga_seq_total = (uint64_t)(frame.fpga_seq_length * (1 - frac_lost));
                frame.rfi_total = (uint64_t)(frame.fpga_seq_length * frac_rfi);
            } else {
                DEBUG("Dropping frame {:d} with frequency ID {:d}.", frame_id_in, frame.freq_id);
                mark_frame_empty(buf_in, unique_name.c_str(), frame_id_in);
                frame_id_in = (frame_id_in + 1) % buf_in->num_frames;
                continue;
            }
        }

        // Mark output frame full and input frame empty
        mark_frame_full(buf_out, unique_name.c_str(), frame_id_out);
        mark_frame_empty(buf_in, unique_name.c_str(), frame_id_in);
        // Move forward one frame
        frame_id_out = (frame_id_out + 1) % buf_out->num_frames;
        frame_id_in = (frame_id_in + 1) % buf_in->num_frames;
    }
}
