#include "bufferCopy.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO, DEBUG2, FATAL_ERROR
#include "visUtil.hpp"

#include "fmt.hpp" // for format, fmt
#include "json.hpp"

#include <algorithm>  // for max
#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdexcept>  // for runtime_error, invalid_argument
#include <stdint.h>   // for uint8_t
#include <string.h>   // for memcpy


using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferCopy);

bufferCopy::bufferCopy(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferCopy::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    _copy_metadata = config.get_default<bool>(unique_name, "copy_metadata", false);

    json buffer_list = config.get<std::vector<json>>(unique_name, "out_bufs");
    Buffer* out_buf = nullptr;
    std::string internal_name;
    std::string buffer_name;
    INFO("buffer_list {:s}", buffer_list.dump());
    for (json buffer : buffer_list) {

        if (buffer.is_object()) {
            // Assuming each array entry has only one key.
            json::iterator it = buffer.begin();
            internal_name = it.key();
            buffer_name = it.value().get<std::string>();
            out_buf = buffer_container.get_buffer(buffer_name);
            if (out_buf == nullptr)
                throw std::runtime_error(
                    fmt::format(fmt("Output buffer is null: {:s}"), buffer.dump()));
        } else if (buffer.is_string()) {
            buffer_name = buffer.get<std::string>();
            internal_name = "";
            out_buf = buffer_container.get_buffer(buffer_name);
            if (out_buf == nullptr)
                throw std::runtime_error(
                    fmt::format(fmt("Output buffer is null: {:s}"), buffer.dump()));
        } else {
            throw std::runtime_error(
                fmt::format(fmt("Unknown value in out_bufs: {:s}"), buffer.dump()));
        }

        if (out_buf->frame_size != in_buf->frame_size) {
            throw std::invalid_argument(fmt::format(fmt("Input buffer '{:s}' not equal to output "
                                                        "buffer size."),
                                                    buffer_name));
        }

        out_buf->register_producer(unique_name);
        INFO("Adding buffer: {:s}:{:s}", internal_name, out_buf->buffer_name);
        out_bufs.push_back(std::make_tuple(internal_name, out_buf, frameID(out_buf)));
    }
}

void bufferCopy::main_thread() {

    frameID in_frame_id(in_buf);

    while (!stop_thread) {
        uint8_t* input_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (input_frame == nullptr)
            break;

        for (auto& buffer_info : out_bufs) {

            const std::string& internal_buffer_name = std::get<0>(buffer_info);
            Buffer* out_buf = std::get<1>(buffer_info);
            frameID& out_frame_id = std::get<2>(buffer_info);

            if (out_buf->get_num_producers() != 1) {
                FATAL_ERROR("Cannot copy into buffer: {:s} as it has more than one producer.",
                            internal_buffer_name);
                return;
            }

            /// Wait for an output frame
            DEBUG2("Waiting for {:s}[{:d}]", out_buf->buffer_name, out_frame_id);
            uint8_t* output_frame =
                wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
            if (output_frame == nullptr)
                goto exit_loop; // Shutdown condition

            // Either make a deep copy or pass the metadata depending if the flag is set
            if (get_metadata_container(in_buf, in_frame_id) != nullptr) {
                if (_copy_metadata) {
                    allocate_new_metadata_object(out_buf, out_frame_id);
                    copy_metadata(in_buf, in_frame_id, out_buf, out_frame_id);
                } else
                    pass_metadata(in_buf, in_frame_id, out_buf, out_frame_id);
            }

            // Copy the frame.
            std::memcpy(output_frame, input_frame, in_buf->frame_size);

            mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
            out_frame_id++;
        }

        // We always release the input buffer even if it isn't selected.
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);

        // Increase the in_frame_id for the input buffer
        in_frame_id++;
    }
exit_loop:;
}
