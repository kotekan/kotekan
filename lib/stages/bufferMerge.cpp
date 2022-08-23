#include "bufferMerge.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, get_num_consumers, get_num_producers, mark_frame_...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, DEBUG2, FATAL_ERROR
#include "visUtil.hpp"         // for frameID, current_time, double_to_ts, modulo

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json, basic_json<>::iterator, basic_json<>::object_t, bas...

#include <algorithm>  // for max
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstring>    // for memcpy
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error, invalid_argument

using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferMerge);

bufferMerge::bufferMerge(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferMerge::main_thread, this)) {

    _timeout = config.get_default<double>(unique_name, "timeout", -1.0);
    _force_copy = config.get_default<bool>(unique_name, "force_copy", false);

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    json buffer_list = config.get<std::vector<json>>(unique_name, "in_bufs");
    Buffer* in_buf = nullptr;
    std::string internal_name;
    std::string buffer_name;
    INFO("buffer_list {:s}", buffer_list.dump());
    for (json buffer : buffer_list) {

        if (buffer.is_object()) {
            // Assuming each array entry has only one key.
            json::iterator it = buffer.begin();
            internal_name = it.key();
            buffer_name = it.value().get<std::string>();
            in_buf = buffer_container.get_buffer(buffer_name);
            assert(in_buf != nullptr);
        } else if (buffer.is_string()) {
            buffer_name = buffer.get<std::string>();
            internal_name = "";
            in_buf = buffer_container.get_buffer(buffer_name);
            assert(in_buf != nullptr);
        } else {
            throw std::runtime_error(
                fmt::format(fmt("Unknown value in in_bufs: {:s}"), buffer.dump()));
        }

        if (in_buf->frame_size != out_buf->frame_size) {
            throw std::invalid_argument(fmt::format(fmt("Input buffer '{:s}' not equal to output "
                                                        "buffer size."),
                                                    buffer_name));
        }

        register_consumer(in_buf, unique_name.c_str());
        INFO("Adding buffer: {:s}:{:s}", internal_name, in_buf->buffer_name);
        in_bufs.push_back(std::make_tuple(internal_name, in_buf, frameID(in_buf)));
    }
}

bool bufferMerge::select_frame(const std::string& internal_name, Buffer* in_buf,
                               uint32_t frame_id) {
    (void)internal_name;
    (void)in_buf;
    (void)frame_id;
    return true;
}

void bufferMerge::main_thread() {

    frameID out_frame_id(out_buf);

    if (get_num_producers(out_buf) != 1) {
        FATAL_ERROR("Cannot merge into a buffer with more than one producer");
        return;
    }

    // If any in buffer has more than one consumer force a copy.
    for (auto& buffer_info : in_bufs) {
        Buffer* in_buf = std::get<1>(buffer_info);
        if (get_num_consumers(in_buf) > 1) {
            WARN("Using a deep copy in {:s} because {:s} has {:d} consumers", unique_name,
                 in_buf->buffer_name, get_num_consumers(in_buf));
            _force_copy = true;
        }
    }

    while (!stop_thread) {
        for (auto& buffer_info : in_bufs) {
            const std::string& internal_buffer_name = std::get<0>(buffer_info);
            Buffer* in_buf = std::get<1>(buffer_info);
            frameID& in_frame_id = std::get<2>(buffer_info);

            /// Wait for an input frame
            if (_timeout < 0) {
                DEBUG2("Waiting for {:s}[{:d}]", in_buf->buffer_name, in_frame_id);
                uint8_t* input_frame =
                    wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
                if (input_frame == nullptr)
                    goto exit_loop; // Shutdown condition
            } else {
                auto timeout = double_to_ts(current_time() + _timeout);
                int status =
                    wait_for_full_frame_timeout(in_buf, unique_name.c_str(), in_frame_id, timeout);
                if (status == 1)
                    continue;
                if (status == -1)
                    goto exit_loop; // Got shutdown signal
            }

            if (select_frame(internal_buffer_name, in_buf, in_frame_id)) {

                uint8_t* output_frame =
                    wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
                if (output_frame == nullptr)
                    break;

                // Move the metadata over to the new frame
                pass_metadata(in_buf, in_frame_id, out_buf, out_frame_id);

                // Copy or swap the frame.
                if (_force_copy) {
                    DEBUG("Doing deep copy on buffer {:s}", in_buf->buffer_name);
                    std::memcpy(output_frame, in_buf->frames[in_frame_id], in_buf->frame_size);
                } else {
                    swap_frames(in_buf, in_frame_id, out_buf, out_frame_id);
                }

                mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
                out_frame_id++;
            }

            // We always release the input buffer even if it isn't selected.
            mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);

            // Increase the in_frame_id for the input buffer
            in_frame_id++;
        }
    }
exit_loop:;
}
