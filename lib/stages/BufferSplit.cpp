#include "BufferSplit.hpp"

#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"       // for mark_frame_empty, mark_frame_full, pass_metadata, register_c...
#include "visUtil.hpp"      // for frameID, modulo

#include <algorithm> // for max
#include <atomic>    // for atomic_bool
#include <memory>    // for allocator_traits<>::value_type
#include <stdint.h>  // for uint8_t, uint32_t

REGISTER_KOTEKAN_STAGE(BufferSplit);

STAGE_CONSTRUCTOR(BufferSplit) {
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_bufs = get_buffer_array("out_bufs");
    for (struct Buffer* out_buf : out_bufs)
        register_producer(out_buf, unique_name.c_str());
}

BufferSplit::~BufferSplit() {}

void BufferSplit::main_thread() {

    std::vector<frameID> output_frame_ids;
    for (auto& out_buf : out_bufs)
        output_frame_ids.push_back(frameID(out_buf));

    frameID input_frame_id(in_buf);

    while (!stop_thread) {
        for (uint32_t i = 0; i < out_bufs.size(); ++i) {
            uint8_t* input = wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id);
            if (input == nullptr)
                break;
            uint8_t* output =
                wait_for_empty_frame(out_bufs.at(i), unique_name.c_str(), output_frame_ids.at(i));
            if (output == nullptr)
                break;

            safe_swap_frame(in_buf, input_frame_id, out_bufs.at(i), output_frame_ids.at(i));

            pass_metadata(in_buf, input_frame_id, out_bufs.at(i), output_frame_ids.at(i));

            mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id++);
            mark_frame_full(out_bufs.at(i), unique_name.c_str(), output_frame_ids.at(i)++);
        }
    }
}
