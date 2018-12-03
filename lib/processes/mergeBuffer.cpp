#include "mergeBuffer.hpp"
#include "json.hpp"
#include "visUtil.hpp"

using nlohmann::json;

REGISTER_KOTEKAN_PROCESS(mergeBuffer);

mergeBuffer::mergeBuffer(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&mergeBuffer::main_thread, this)) {

    _timeout = config.get_default<double>(unique_name, "timeout", -1.0);

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    json buffer_list = config.get<std::vector<json>>(unique_name, "in_bufs");
    Buffer * in_buf = nullptr;
    INFO("buffer_list %s", buffer_list.dump().c_str());
    for (json buffer : buffer_list) {
        // Assuming each array entry has only one key.
        json::iterator it = buffer.begin();
        std::string internal_name = it.key();
        std::string buffer_name = it.value();
        in_buf = buffer_container.get_buffer(buffer_name);
        assert(in_buf != nullptr);

        if(in_buf->frame_size != out_buf->frame_size) {
            throw std::invalid_argument("Input buffer '" + buffer_name +
                                        "' not equal to output buffer size.");
        }

        register_consumer(in_buf, unique_name.c_str());
        INFO("Adding buffer: %s:%s", internal_name.c_str(), in_buf->buffer_name);
        in_bufs.push_back(std::make_tuple(internal_name, in_buf, 0));
    }

}

bool mergeBuffer::select_frame(const std::string &internal_name,
                               Buffer * in_buf, uint32_t frame_id) {
    (void)internal_name;
    (void)in_buf;
    (void)frame_id;
    return true;
}

void mergeBuffer::main_thread() {

    Buffer * in_buf;
    uint32_t in_frame_id;
    std::string internal_buffer_name;

    uint32_t out_frame_id = 0;

    while(!stop_thread) {
        for (auto &buffer_info : in_bufs) {
            std::tie(internal_buffer_name, in_buf, in_frame_id) = buffer_info;

            /// Wait for an input frame
            if (_timeout < 0) {
                DEBUG2("Waiting for %s[%d]", in_buf->buffer_name, in_frame_id);
                uint8_t * input_frame = wait_for_full_frame(in_buf, unique_name.c_str(),
                                                            in_frame_id);
                if (input_frame == NULL) goto exit_loop; // Shutdown condition
            } else {
                auto timeout = double_to_ts(current_time() + _timeout);
                int status = wait_for_full_frame_timeout(in_buf, unique_name.c_str(),
                                                         in_frame_id, timeout);
                if(status == 1) continue;
                if(status == -1) goto exit_loop;  // Got shutdown signal
            }

            if (select_frame(internal_buffer_name, in_buf, in_frame_id)) {

                uint8_t * output_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
                if (output_frame == NULL) break;

                swap_frames(in_buf, in_frame_id, out_buf, out_frame_id);
                pass_metadata(in_buf, in_frame_id, out_buf, out_frame_id);

                mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
                out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
            }

            // We always release the input buffer even if it isn't selected.
            mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);

            std::get<2>(buffer_info) = (in_frame_id + 1) % in_buf->num_frames;
        }
    }
    exit_loop:;
}
