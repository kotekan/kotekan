#include <stdio.h>
#include <thread>
#include <assert.h>

#include "basebandReadout.hpp"
#include "baseband_manager.hpp"
#include "buffer.h"
#include "errors.h"
#include "fpga_header_functions.h"


REGISTER_KOTEKAN_PROCESS(basebandReadout);


/// Worker task that mocks the progress of a baseband dump
// TODO: implement
static void process_request(const std::shared_ptr<BasebandDump> dump) {
    std::cout << "Started processing " << dump << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    dump->bytes_remaining -= 51;
    std::cout << "Half way processing " << dump << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    dump->bytes_remaining = 0;
    std::cout << "Finished processing " << dump << std::endl;
}


basebandReadout::basebandReadout(Config& config, const string& unique_name,
                                 bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&basebandReadout::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    base_dir = config.get_string_default(unique_name, "base_dir", "./");
    file_ext = config.get_string(unique_name, "file_ext");
    num_frames_buffer = config.get_int(unique_name, "num_frames_buffer");

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
    manager->num_elements = config.get_int(unique_name, "num_elements");
    manager->samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
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
    uint64_t event_id=0;
    BasebandManager& mgr = BasebandManager::instance();

    while (!stop_thread) {
        // Code that listens and waits for triggers and fills in trigger parameters.
        // Latency is *key* here. We want to call manager->get_data within 100ms
        // of L4 sending the trigger.

        // Code to run after getting a trigger.

        // For testing readout logic.
        //auto dump = mgr.get_next_dump();
        auto dump =  std::make_shared<BasebandDump>(
                BasebandDump{BasebandRequest{360000, 131079}});
        sleep(5);

        if (dump) {
            std::cout << "Something to do!" << std::endl;
            std::time_t tt = std::chrono::system_clock::to_time_t(dump->request.received);
            std::cout << "Received: " << std::put_time(std::localtime(&tt), "%F %T")
                      << ", start: " << dump->request.start_fpga
                      << ", length: " << dump->request.length_fpga << std::endl;

            // Copying the data from the ring buffer is done in *this* thread. Writing the data out is done
            // by a new thread. This keeps the number of threads that can lock out the main buffer limited
            // to 2 (listen and main).
            basebandDump data = manager->get_data(
                    event_id,    // XXX need this from the request.
                    dump->request.start_fpga,
                    dump->request.length_fpga
                    );

            // Spawn thread to write out the data.
            dump->bytes_remaining -= 42;
            std::cout << "processing: " << dump
                      << ", saved: " << dump->bytes_remaining
                      << std::endl;
            // Here we also want to limit the number of simultanious readout threads, mostly so
            // we don't run out of memory and crash everything. That will also prevent too
            // many calls to get_data and thus buffer deadlock.
            std::thread worker(process_request, dump);
            worker.detach();
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
    std::lock_guard<std::mutex> lock(manager_lock);
    int replaced_frame = -1;
    assert(frame_id == next_frame);

    // This will block if we are trying to replace a frame currenty being read out.
    frame_locks[frame_id % length].lock();
    // Somehow in C `-1 % length == -1` which makes no sence to me.
    // So add `length` to `oldest_frame`.
    if (frame_id % length == (oldest_frame + length) % length) {
        replaced_frame = oldest_frame;
        oldest_frame++;
    }
    frame_locks[frame_id % length].unlock();

    next_frame++;
    return replaced_frame;
}

bufferManager::~bufferManager() {
}

basebandDump bufferManager::get_data(
        uint64_t event_id,
        int64_t trigger_start_fpga,
        int64_t trigger_length_fpga
        ) {
    manager_lock.lock();

    int dump_start_frame = (oldest_frame > 0) ? oldest_frame : 0;
    int dump_end_frame = dump_start_frame;



    std::cout << "Dump samples: " << trigger_start_fpga << " : " << trigger_start_fpga + trigger_length_fpga << std::endl;

    for (int frame_index = dump_start_frame; frame_index < next_frame; frame_index++) {
        int buf_frame = frame_index % buf->num_frames;
        chimeMetadata metadata = *((chimeMetadata *) buf->metadata[buf_frame]->metadata);
        int64_t frame_fpga_seq = metadata.fpga_seq_num;
        std::cout << buf_frame << " : " << frame_fpga_seq << std::endl;
        if (trigger_start_fpga + trigger_length_fpga <= frame_fpga_seq) continue;
        if (trigger_start_fpga >= frame_fpga_seq + samples_per_data_set) {
            dump_start_frame = frame_index + 1;
            continue;
        }
        frame_locks[frame_index % length].lock();
        dump_end_frame = frame_index + 1;
    }

    // Now that the relevant frames are locked, we can unlock the manager so the
    // rest of the buffer can continue to opperate.
    manager_lock.unlock();

    std::cout << "Frames in dump: " << dump_start_frame << " : " << dump_end_frame << std::endl;

    uint32_t freq_id = 10; //fake
    int64_t data_length_fpga = 10000;
    int64_t data_start_fpga = 0;

    // Figure out how much data we have.

    basebandDump dump(
            event_id,
            freq_id,
            num_elements,
            data_start_fpga,
            data_length_fpga
            );

    // Fill in the data.
    for (int ii = 0; ii < data_length_fpga; ii++) {
        for (int jj = 0; jj < num_elements; jj++) {
            dump.data[ii * num_elements + jj] = 8;
        }
    }

    std::cout << "Here." << std::endl;

    // All the data has been copied. Release the locks.
    for (int frame_index = dump_start_frame; frame_index < dump_end_frame; frame_index++) {
        frame_locks[frame_index % length].unlock();
    }

    return dump;
}


template<typename T>
gsl::span<T> span_from_length(uint8_t * start, size_t length) {
    T* span_start = (T*)start;
    T* span_end = (T*)(start + length);

    return gsl::span<T>(span_start, span_end);
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
        // XXX Since I didn't supply a deleter, I think it is implicitly `delete`
        // instead of `delete[]`, so this should leak memory.
        // Actually this does not appear to leak. Not positive I understand why.
        data_ref(new uint8_t[num_elements_ * data_length_fpga_])
        // Candidate correct constructor.
        // data_ref(new uint8_t[num_elements_ * data_length_fpga_],
        //      [](uint8_t *p ) { delete[] p; }),
        // Intensional leak, for testing.
        // data_ref(new uint8_t[num_elements_ * data_length_fpga_],
        //      [](uint8_t *p ) { std::cout << "Delete!" << std::endl; })

        // This appearently isn't allowed since data_ref hasn't been inialized.
        //data(data_ref.get(), data_ref.get() + num_elements_ * data_length_fpga_)

        // If we initialized the memory in the span instead of the ref.
        // data(span_from_length<uint8_t>(new uint8_t[num_elements_ * data_length_fpga_],
        //                               num_elements_ * data_length_fpga_))
{

    data = gsl::span<uint8_t>(data_ref.get(),
                              data_ref.get() + num_elements * data_length_fpga);

    //data_ref.reset(&(data[0]));
}

basebandDump::~basebandDump() {
}
