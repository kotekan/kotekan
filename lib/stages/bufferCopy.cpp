#include "bufferCopy.hpp"

#include "visUtil.hpp"

#include "json.hpp"

#include <signal.h>

using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferCopy);

bufferCopy::bufferCopy(Config& config, const string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferCopy::main_thread, this)) {

    _timeout = config.get_default<double>(unique_name, "timeout", -1.0);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

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
            assert(out_buf != nullptr);
        } else if (buffer.is_string()) {
            buffer_name = buffer.get<std::string>();
            internal_name = "";
            out_buf = buffer_container.get_buffer(buffer_name);
            assert(out_buf != nullptr);
        } else {
            throw std::runtime_error(
                fmt::format(fmt("Unknown value in out_bufs: {:s}"), buffer.dump()));
        }

        if (out_buf->frame_size != in_buf->frame_size) {
            throw std::invalid_argument(fmt::format(fmt("Input buffer '{:s}' not equal to output "
                                                        "buffer size."),
                                                    buffer_name));
        }

        register_producer(out_buf, unique_name.c_str());
        INFO("Adding buffer: {:s}:{:s}", internal_name, out_buf->buffer_name);
        out_bufs.push_back(std::make_tuple(internal_name, out_buf, frameID(out_buf)));
    }
}

bool bufferCopy::select_frame(const std::string& internal_name, Buffer* out_buf,
                               uint32_t frame_id) {
    (void)internal_name;
    (void)out_buf;
    (void)frame_id;
    return true;
}

void bufferCopy::main_thread() {

    frameID in_frame_id(in_buf);

    //if (get_num_producers(out_buf) != 1) {
    //    FATAL_ERROR("Cannot merge into a buffer with more than one producer");
    //    return;
    //}

    while (!stop_thread) {
      uint8_t* input_frame =
        wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
      if (input_frame == NULL)
        break;

      for (auto& buffer_info : out_bufs) {
        const std::string& internal_buffer_name = std::get<0>(buffer_info);
        Buffer* out_buf = std::get<1>(buffer_info);
        frameID& out_frame_id = std::get<2>(buffer_info);

        /// Wait for an output frame
        DEBUG2("Waiting for {:s}[{:d}]", out_buf->buffer_name, out_frame_id);
        uint8_t* output_frame =
          wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
        if (output_frame == NULL)
          goto exit_loop; // Shutdown condition

        if (select_frame(internal_buffer_name, in_buf, in_frame_id)) {

          // Move the metadata over to the new frame
          pass_metadata(in_buf, in_frame_id, out_buf, out_frame_id);

          // Copy the frame.
          std::memcpy(output_frame, input_frame, in_buf->frame_size);

          mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
          out_frame_id++;
        }
      }

      // We always release the input buffer even if it isn't selected.
      mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);

      // Increase the in_frame_id for the input buffer
      in_frame_id++;

    }
exit_loop:;
}
