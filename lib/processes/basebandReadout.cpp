#include <stdio.h>
#include <thread>
#include <assert.h>

#include "basebandReadout.hpp"
#include "buffer.h"
#include "errors.h"
#include "fpga_header_functions.h"


REGISTER_KOTEKAN_PROCESS(basebandReadout);


basebandReadout::basebandReadout(Config& config, const string& unique_name,
                                 bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&basebandReadout::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    base_dir = config.get_string_default(unique_name, "base_dir", "./");
    file_ext = config.get_string(unique_name, "file_ext");
    num_frames_buffer = config.get_int(unique_name, "num_frames_buffer");
    num_elements = config.get_int(unique_name, "num_elements");
    samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    // XXX
    std::cout << "BB constructor " << num_frames_buffer << " " << buf->num_frames << std::endl;

    // Ensure input buffer is long enough.
    if (buf->num_frames <= num_frames_buffer) {
        // This process of creating an error string is rediculous. Figure out what
        // the std::string way to do this is.
        const int msg_len = 200;
        char msg[200];
        snprintf(msg, msg_len,
                 "Input buffer (%d frames) not large enough to buffer %d frames",
                 buf->num_frames, num_frames_buffer);
        throw std::runtime_error(msg);
    }

    manager = new bufferManager(buf, num_frames_buffer);
}

basebandReadout::~basebandReadout() {
    delete(manager);
}

void basebandReadout::apply_config(uint64_t fpga_seq) {
}

void basebandReadout::main_thread() {

    int frame_id = 0;
    int done_frame;

    std::thread lt(&basebandReadout::listen_thread, this);

    while (!stop_thread) {

        if (wait_for_full_frame(buf, unique_name.c_str(),
                                frame_id % buf->num_frames) == nullptr) {
            break;
        }
        done_frame = manager->add_replace_frame(frame_id);
        if (done_frame >= 0) {
            mark_frame_empty(buf, unique_name.c_str(),
                             done_frame % buf->num_frames);
        }

        std::cout << "Discard: " << done_frame << ", add " << frame_id << std::endl;

        frame_id++;
    }
    lt.join();
}

void basebandReadout::listen_thread() {
    int64_t trigger_start_fpga=0, trigger_length_fpga=0;
    uint64_t event_id=0;

    while (!stop_thread) {
        // Code that listens and waits for triggers and fills in trigger parameters.
        // Latency is *key* here. We want to call manager->get_data within 100ms
        // of L4 sending the trigger.
        sleep(1);
        std::cout << "I'm waiting." << std::endl;
        // Code to run after getting a trigger.
        if (0) {
            basebandDump data = manager->get_data(
                    event_id,
                    trigger_start_fpga,
                    trigger_length_fpga
                    );
            // Spawn thread to write out the data.
        }
        // Somehow keep track of active writer threads, clean them up, and free any memory.
    }
    // Make sure all writer threads are done.
}


bufferManager::bufferManager(Buffer * buf_, int length_) :
    buf(buf_), length(length_), next_frame(0), oldest_frame(-1),
    frame_locks(length), manager_lock() {
}

int bufferManager::add_replace_frame(int frame_id) {
    manager_lock.lock();
    int replaced_frame = -1;
    assert(frame_id == next_frame);

    // This will block if we are trying to replace a frame currenty being read out.
    frame_locks[frame_id % length].lock();
    // Somehow in C `-1 % length == -1` which makes no sence to me.
    if (frame_id % length == (oldest_frame + length) % length) {
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

basebandDump bufferManager::get_data(
        uint64_t event_id,
        int64_t trigger_start_fpga,
        int64_t trigger_length_fpga
        ) {
    uint32_t freq_id = 10; //fake
    int64_t data_length_fpga = 0;
    int64_t data_start_fpga = 0;
    uint32_t num_elements = 0;

    // Figure out how much data we have.

    basebandDump dump(
            event_id,
            freq_id,
            num_elements,
            data_start_fpga,
            data_start_fpga
            );

    // Fill in the data.

    return dump;
}


basebandDump::basebandDump(
        uint64_t event_id_,
        uint32_t freq_id_,
        uint32_t num_elements_,
        int64_t data_start_fpga_,
        int64_t data_length_fpga_
        ) :
        event_id(event_id_),
        freq_id(freq_id_),
        num_elements(num_elements_),
        data_start_fpga(data_start_fpga_),
        data_length_fpga(data_length_fpga_),
        data(new uint8_t[num_elements_ * data_length_fpga_])
        {
}

basebandDump::~basebandDump() {
}
