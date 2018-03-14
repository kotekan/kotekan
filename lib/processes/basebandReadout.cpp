#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <thread>

#include "basebandReadout.hpp"
#include "buffer.h"
#include "errors.h"

REGISTER_KOTEKAN_PROCESS(basebandReadout);

basebandReadout::basebandReadout(Config& config,
                 const string& unique_name,
                 bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&basebandReadout::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    base_dir = config.get_string_default(unique_name, "base_dir", "./");
    file_ext = config.get_string(unique_name, "file_ext");
    num_frames_buffer = config.get_int(unique_name, "num_frames_buffer");

    // XXX
    std::cout << "YOYOYO  " << num_frames_buffer << " " << buf->num_frames << std::endl;

    // Ensure input buffer is long enough.
    if (buf->num_frames <= num_frames_buffer) {
        const int msg_len = 200;
        char msg[200];
        snprintf(msg, msg_len,
                 "Input buffer (%d frames) not large enough to buffer %d frames",
                 buf->num_frames, num_frames_buffer);
        throw std::runtime_error(msg);
    }

}

basebandReadout::~basebandReadout() {
}

void basebandReadout::apply_config(uint64_t fpga_seq) {
}

void basebandReadout::main_thread() {

    int fd;
    int file_num = 0;
    int frame_id = 0;
    uint8_t * frame = NULL;
    char hostname[64];
    gethostname(hostname, 64);

    bufferManager manager(num_frames_buffer);


    while (!stop_thread) {

        std::cout << "START THREAD" << std::endl;

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        const int full_path_len = 200;
        char full_path[full_path_len];

        snprintf(full_path, full_path_len, "%s/%s_%07d.%s",
                base_dir.c_str(),
                hostname,
                file_num,
                file_ext.c_str());

        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = ( frame_id + 1 ) % buf->num_frames;
        file_num++;
    }
}


bufferManager::bufferManager(unsigned int length_) :
    length(length_), next_frame(0), oldest_frame(-1), metas(length, NULL), 
    frames(length, NULL), manager_lock() {
}

bufferManager::~bufferManager() {
}
