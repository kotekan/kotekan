#include "restInspectFrame.hpp"

REGISTER_KOTEKAN_PROCESS(restInspectFrame);

restInspectFrame::restInspectFrame(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&restInspectFrame::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf_config_name = config.get_string(unique_name, "in_buf");
    register_consumer(in_buf, unique_name.c_str());

    len = config.get_int_default(unique_name, "len", 0);

    if (len == 0) {
        len = in_buf->frame_size;
    } else if (len > in_buf->frame_size) {
        WARN("Requested len (%d) is greater than the frame_size (%d).",
             len, in_buf->frame_size);
        len = in_buf->frame_size;
    }

    frame_copy = (uint8_t *)malloc(len);
    CHECK_MEM(frame_copy);
}

restInspectFrame::~restInspectFrame() {
    free(frame_copy);
}

void restInspectFrame::apply_config(uint64_t fpga_seq) {}

void restInspectFrame::rest_callback(connectionInstance &conn) {
    frame_copy_lock.lock();
    conn.send_binary_reply(frame_copy, len);
    frame_copy_lock.unlock();
}

void restInspectFrame::main_thread() {

    uint8_t * frame = NULL;
    uint32_t frame_id = 0;
    bool registered = false;

    while(!stop_thread) {
        frame = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        if (frame_copy_lock.try_lock()) {
            // TODO Enforce alignemnt needed to use nt_memcpy() here.
            memcpy((void*)frame_copy, (void*)frame, len);
            frame_copy_lock.unlock();
        }

        // Only register the callback once we have something to return
        if (!registered) {
            using namespace std::placeholders;
            restServer * rest_server = get_rest_server();
            std::string endpoint = "/inspect_frame/" + in_buf_config_name;
            rest_server->register_get_callback(endpoint,
                std::bind(&restInspectFrame::rest_callback, this, _1));
            registered = true;
        }

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}