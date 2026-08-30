#include "bufferDedup.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp" // for FATAL_ERROR, DEBUG
#include "visUtil.hpp"        // for frameID, modulo

#include "fmt.hpp" // for compile_string_to_view

#include <cstring>    // for memcmp, memcpy
#include <functional> // for bind, function

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferDedup);

bufferDedup::bufferDedup(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferDedup::main_thread, this)),
    in_buf(get_buffer("in_buf")), out_buf(get_buffer("out_buf")),
    resend_after_frames(config.get_default<int>(unique_name, "resend_after_frames", 0)) {

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);

    if (in_buf->frame_size != out_buf->frame_size)
        FATAL_ERROR("in_buf frame size ({:d}) must equal out_buf frame size ({:d})",
                    in_buf->frame_size, out_buf->frame_size);
}

void bufferDedup::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);
    int suppressed = 0;

    while (!stop_thread) {
        const uint8_t* frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (frame == nullptr)
            break;

        const bool changed =
            last_sent.empty() || std::memcmp(last_sent.data(), frame, in_buf->frame_size) != 0;
        const bool resend = resend_after_frames > 0 && suppressed >= resend_after_frames;

        if (changed || resend) {
            uint8_t* out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
            if (out_frame == nullptr)
                break;

            if (in_buf->get_metadata(in_frame_id))
                in_buf->pass_metadata(in_frame_id, out_buf, out_frame_id);
            std::memcpy(out_frame, frame, in_buf->frame_size);
            last_sent.assign(frame, frame + in_buf->frame_size);

            DEBUG("Forwarding {:s} frame {:d} ({:s})", in_buf->buffer_name, in_frame_id,
                  changed ? "changed" : "resend");
            out_buf->mark_frame_full(unique_name, out_frame_id++);
            suppressed = 0;
        } else {
            suppressed++;
        }

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
    }
}
