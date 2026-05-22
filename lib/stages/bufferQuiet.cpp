#include "bufferQuiet.hpp"

#include <time.h>              // for time_t, timespec
#include <assert.h>            // for assert
#include <bits/chrono.h>       // for duration, time_point, time_point_cast, nanoseconds, operator+
#include <fmt/core.h>          // for format, format_string
#include <stdint.h>            // for uint8_t
#include <cstring>             // for memcpy
#include <functional>          // for bind, function
#include <stdexcept>           // for runtime_error
#include <memory>              // for shared_ptr, __shared_ptr_access
#include <vector>              // for vector

#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "kotekanLogging.hpp"  // for DEBUG
#include "metadata.hpp"        // for metadataObject
#include "visUtil.hpp"         // for frameID, modulo
#include "fmt.hpp"             // for compile_string_to_view, fmt


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferQuiet);

bufferQuiet::bufferQuiet(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferQuiet::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Frame sizes must match exactly
    if (in_buf->frame_size != out_buf->frame_size) {
        throw std::runtime_error(
            fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                        in_buf->frame_size, out_buf->frame_size));
    }

    _copy_frame = config.get_default<bool>(unique_name, "copy_frame", false);
    _quiet_time = config.get<double>(unique_name, "quiet_time");
    _num_reserve_frames = config.get_default<int>(unique_name, "num_reserve_frames", 0);

    // Frame sizes must match exactly
    if (in_buf->num_frames < _num_reserve_frames) {
        throw std::runtime_error(fmt::format(
            "Cannot reserve more ({:d}) frames than present in input buffer {:s} (:{d}).",
            _num_reserve_frames, in_buf->buffer_name, in_buf->num_frames));
    }
}

void bufferQuiet::main_thread() {

    frameID in_frame_id(in_buf);
    frameID in_frame_release_id(in_buf);
    frameID out_frame_id(out_buf);
    int in_frame_hold_ctr = 0;

    uint8_t* output_frame = nullptr;

    bool first_time = true;
    while (!stop_thread) {

        // system_clock is the time used by wait_for_full_frame_timeout
        const auto now = std::chrono::time_point_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now());
        // we (effectively) disable timeout for first frame, to work around
        // startup delays of incoming pipeline
        const auto timeout_point_nsecs =
            now
            + std::chrono::nanoseconds(time_t((first_time ? 365 * 24 * 3600 : _quiet_time) * 1e9));
        // conversion to timespec is somewhat complicated
        const auto timeout_point_secs =
            std::chrono::time_point_cast<std::chrono::nanoseconds>(timeout_point_nsecs);
        const auto ns =
            timeout_point_nsecs
            - std::chrono::time_point_cast<std::chrono::nanoseconds>(timeout_point_secs);
        const timespec timeout = {timeout_point_secs.time_since_epoch().count(), ns.count()};

        const int reason = in_buf->wait_for_full_frame_timeout(unique_name, in_frame_id, timeout);
        if (reason == -1) // thread exit signal
            return;

        if (reason == 0) { // got a frame before timeout
            DEBUG("Got new input frame {:d}", in_frame_id);
            in_frame_hold_ctr += 1;
            in_frame_id += 1;
        }

        int frames_to_release = -1;
        if (reason == 1) { // timeout, release all frames
            frames_to_release = in_frame_hold_ctr;
        } else if (in_frame_hold_ctr + _num_reserve_frames
                   >= in_buf->num_frames) { // emergency release a frame since buffer is full
            assert(reason == 0);
            assert(in_frame_hold_ctr + _num_reserve_frames
                   == in_buf->num_frames); // we count up be ones, we cannot miss equality
            frames_to_release = 1;
        } else { // got a frame, no timeout, no full buffer. Wait.
            assert(reason == 0);
            frames_to_release = 0;
        }

        DEBUG("Holding {:d} frames, {:d} frames set to be released", in_frame_hold_ctr,
              frames_to_release);

        while (frames_to_release-- > 0) {

            // Get a new output frame
            output_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
            if (output_frame == nullptr)
                return;

            if (_copy_frame || in_buf->get_num_consumers() > 1) {
                out_buf->allocate_new_metadata_object(out_frame_id);
                // Metadata sizes must match exactly
                if (in_buf->metadata[in_frame_release_id]->get_object_size()
                    != out_buf->metadata[out_frame_id]->get_object_size()) {
                    throw std::runtime_error(fmt::format(
                        fmt("Metadata sizes must match for direct copy (src {:d} != dest {:d})."),
                        in_buf->metadata[in_frame_release_id]->get_object_size(),
                        out_buf->metadata[out_frame_id]->get_object_size()));
                }
                in_buf->copy_metadata(in_frame_release_id, out_buf, out_frame_id);
                std::memcpy(output_frame, in_buf->frames[in_frame_release_id], in_buf->frame_size);
                // this just copies the shared pointer, and is only valid since
                // a frame description must never change
            } else {
                in_buf->pass_metadata(in_frame_release_id, out_buf, out_frame_id);
                in_buf->swap_frames(in_frame_release_id, out_buf, out_frame_id);
                // this just copies the shared pointer, and is only valid since
                // a frame description must never change
            }
            const auto in_frame_desc = in_buf->get_frame_description();
            if (in_frame_desc) {
                out_buf->set_frame_desc(in_frame_desc);
            } else {
                assert(!out_buf->get_frame_description());
            }

            DEBUG("Releasing frame... in_frame_id: {:d}, in_frame_hold_ctr: {:d}, "
                  "in_frame_release_id: {:d}",
                  in_frame_id, in_frame_hold_ctr, in_frame_release_id);

            out_buf->mark_frame_full(unique_name, out_frame_id);
            out_frame_id++;

            // Release the input buffer frame.
            in_buf->mark_frame_empty(unique_name, in_frame_release_id);
            in_frame_release_id++;

            in_frame_hold_ctr--;
        }

        first_time = false;
    }
}
