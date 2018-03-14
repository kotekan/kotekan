#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <thread>
#include <assert.h>

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

    int frame_id = 0;
    int done_frame;
    uint8_t * frame = NULL;

    bufferManager manager(buf, num_frames_buffer);

    std::cout << "START LOOP" << std::endl;

    while (!stop_thread) {

        frame = wait_for_full_frame(buf, unique_name.c_str(),
                                    frame_id % buf->num_frames);
        done_frame = manager.add_replace_frame(frame_id);
        if (done_frame >= 0) {
            mark_frame_empty(buf, unique_name.c_str(),
                             done_frame % buf->num_frames);
        }

        std::cout << frame_id << " : " << done_frame << std::endl;

        frame_id++;
    }
}


bufferManager::bufferManager(Buffer * buf_, int length_) :
    buf(buf_), length(length_), next_frame(0), oldest_frame(-1),
    frame_locks(length), manager_lock() {
}

int bufferManager::add_replace_frame(int frame_id) {
    manager_lock.lock();
    int replaced_frame = -1;
    assert(frame_id == next_frame);

    // This will block if we are trying to replace a frame currenlty being read out.
    frame_locks[frame_id % length].lock();
    if (frame_id % length == oldest_frame % length) {
        replaced_frame = oldest_frame;
        oldest_frame++;
    }
    frame_locks[frame_id % length].unlock();

    next_frame++;
    manager_lock.unlock();
    return replaced_frame;
}

bufferManager::~bufferManager() {
}
