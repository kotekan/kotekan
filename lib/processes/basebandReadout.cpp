#include <stdio.h>
#include <thread>
#include <assert.h>
#include <algorithm>
#include <cstdint>

#include "basebandReadout.hpp"
#include "buffer.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "gpsTime.h"
#include "nt_memcpy.h"
#include "nt_memset.h"



REGISTER_KOTEKAN_PROCESS(basebandReadout);


/// Worker task that mocks the progress of a baseband dump
// TODO: implement
static void process_request(const std::shared_ptr<BasebandDumpStatus> dump, basebandDumpData data) {
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
                       std::bind(&basebandReadout::main_thread, this)),
        buf(get_buffer("in_buf")),
        base_dir(config.get_string_default(unique_name, "base_dir", "./")),
        file_ext(config.get_string(unique_name, "file_ext")),
        num_frames_buffer(config.get_int(unique_name, "num_frames_buffer")),
        num_elements(config.get_int(unique_name, "num_elements")),
        samples_per_data_set(config.get_int(unique_name, "samples_per_data_set")),
        next_frame(0),
        oldest_frame(-1),
        frame_locks(num_frames_buffer)
{
    // Memcopy byte alignments assume the following.
    if (num_elements % 128) {
        throw std::runtime_error("num_elements must be multiple of 128");
    }

    register_consumer(buf, unique_name.c_str());

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
}

basebandReadout::~basebandReadout() {
}

void basebandReadout::apply_config(uint64_t fpga_seq) {
}

void basebandReadout::main_thread() {

    int frame_id = 0;
    int done_frame;

    std::thread wt(&basebandReadout::write_thread, this);
    std::unique_ptr<std::thread> lt;

    while (!stop_thread) {

        if (wait_for_full_frame(buf, unique_name.c_str(),
                                frame_id % buf->num_frames) == nullptr) {
            break;
        }

        if (!lt) {
            int buf_frame = frame_id % buf->num_frames;
            auto first_meta = (chimeMetadata *) buf->metadata[buf_frame]->metadata;

            stream_id_t stream_id = extract_stream_id(first_meta->stream_ID);
            uint32_t freq_id = bin_number_chime(&stream_id);
            INFO("Starting request-listening thread for freq_id: %d", freq_id);
            lt = std::make_unique<std::thread>(&basebandReadout::listen_thread, this, freq_id);
        }

        done_frame = add_replace_frame(frame_id);
        if (done_frame >= 0) {
            mark_frame_empty(buf, unique_name.c_str(),
                             done_frame % buf->num_frames);
        }

        // XXX Delete dev prints.
        std::cout << "Discard: " << done_frame << ", add " << frame_id << std::endl;

        frame_id++;
    }

    if (lt) {
        lt->join();
    }
    wt.join();
}

void basebandReadout::listen_thread(const uint32_t freq_id) {
    uint64_t event_id=0;

    BasebandRequestManager& mgr = BasebandRequestManager::instance();

    while (!stop_thread) {
        // Code that listens and waits for triggers and fills in trigger parameters.
        // Latency is *key* here. We want to call get_data within 100ms
        // of L4 sending the trigger.

        auto dump_status = mgr.get_next_request(freq_id);

        if (dump_status) {
            std::cout << "Something to do!" << std::endl;
            std::time_t tt = std::chrono::system_clock::to_time_t(dump_status->request.received);
            std::cout << "Received: " << std::put_time(std::localtime(&tt), "%F %T")
                      << ", start: " << dump_status->request.start_fpga
                      << ", length: " << dump_status->request.length_fpga << std::endl;

            // Copying the data from the ring buffer is done in *this* thread. Writing the data
            // out is done by a new thread. This keeps the number of threads that can lock out
            // the main buffer limited to 2 (listen and main).
            basebandDumpData data = get_data(
                    event_id,    // TODO need this from the request.
                    dump_status->request.start_fpga,
                    dump_status->request.length_fpga
                    );

            // At this point we know how much of the requested data we managed to read from the
            // buffer (which may be nothing if the request as recieved too late). Do we need to
            // report this?
            if (data.data_length_fpga == 0) {
                INFO("Captured no data for event %d and freq ID %d.", data.event_id, data.freq_id);
                // TODO Report to the request.
                continue;
            } else {
                INFO("Captured %d samples for event %d and freq ID %d.",
                     data.data_length_fpga,
                     data.event_id,
                     data.freq_id
                     );
                // TODO Report to the dump_request.
            }

            // Wait for free space in the write queue. This prevents this thread from receiving any
            // more dump requests until the pipe clears out. Limits the memory use and buffer congestion.
            const int max_writes_queued = 3;
            while (!stop_thread) {
                if (write_q.size() < max_writes_queued) {
                    // XXX Delete dev prints.
                    std::cout << "Adding write to queue. q len: " << write_q.size() << std::endl;
                    write_q.push(std::tuple<basebandDumpData, std::shared_ptr<BasebandDumpStatus>>(data,
                                 dump_status));
                    break;
                } else {
                    sleep(1);
                }
            }
        }
    }
}

void basebandReadout::write_thread() {
    while (!stop_thread) {
        if (write_q.empty()) {
            sleep(1);
        } else {
            auto dump_tup = write_q.front();

            auto dump_status = std::get<1>(dump_tup);
            auto data = std::get<0>(dump_tup);

            process_request(dump_status, data);
            // pop at the end such that memory for `data` is released before queue slot
            // becomes available.
            write_q.pop();
        }
    }
}

int basebandReadout::add_replace_frame(int frame_id) {
    std::lock_guard<std::mutex> lock(manager_lock);
    int replaced_frame = -1;
    assert(frame_id == next_frame);

    // This will block if we are trying to replace a frame currenty being read out.
    frame_locks[frame_id % num_frames_buffer].lock();
    // Somehow in C `-1 % num_frames_buffer == -1` which makes no sence to me.
    // So add `num_frames_buffer` to `oldest_frame`.
    if (frame_id % num_frames_buffer == (oldest_frame + num_frames_buffer) % num_frames_buffer) {
        replaced_frame = oldest_frame;
        oldest_frame++;
    }
    frame_locks[frame_id % num_frames_buffer].unlock();

    next_frame++;
    return replaced_frame;
}

basebandDumpData basebandReadout::get_data(
        uint64_t event_id,
        int64_t trigger_start_fpga,
        int64_t trigger_length_fpga
        ) {
    // This assumes that the frame's timestamps are in order, but not that they
    // are nessisarily contiguous.

    int dump_start_frame;
    int dump_end_frame;
    int64_t trigger_end_fpga = trigger_start_fpga + trigger_length_fpga;
    float max_wait_time = 1.;
    float min_wait_time = samples_per_data_set * FPGA_PERIOD_NS * 1e-9;

    if (trigger_length_fpga > samples_per_data_set * num_frames_buffer / 2) {
        throw std::runtime_error("Baseband dump request too long");
    }

    // XXX delete dev prints.
    std::cout << "Dump samples: " << trigger_start_fpga;
    std::cout << " : " << trigger_start_fpga + trigger_length_fpga << std::endl;

    while (!stop_thread) {
        int64_t frame_fpga_seq = -1;
        manager_lock.lock();
        dump_start_frame = (oldest_frame > 0) ? oldest_frame : 0;
        dump_end_frame = dump_start_frame;

        for (int frame_index = dump_start_frame; frame_index < next_frame; frame_index++) {
            int buf_frame = frame_index % buf->num_frames;
            auto metadata = (chimeMetadata *) buf->metadata[buf_frame]->metadata;
            frame_fpga_seq = metadata->fpga_seq_num;

            if (trigger_end_fpga <= frame_fpga_seq) continue;
            if (trigger_start_fpga >= frame_fpga_seq + samples_per_data_set) {
                dump_start_frame = frame_index + 1;
                continue;
            }
            dump_end_frame = frame_index + 1;
        }
        lock_range(dump_start_frame, dump_end_frame);

        // Now that the relevant frames are locked, we can unlock the rest of the buffer so
        // it can continue to opperate.
        manager_lock.unlock();


        // XXX delete dev prints.
        std::cout << "Frames in dump: " << dump_start_frame;
        std::cout << " : " << dump_end_frame << std::endl;

        // Check if the trigger is 'prescient'. That is, if any of the requested data has
        // not yet arrived.
        int64_t last_sample_present = frame_fpga_seq + samples_per_data_set;
        if (last_sample_present <= trigger_start_fpga + trigger_length_fpga) {
            unlock_range(dump_start_frame, dump_end_frame);
            int64_t time_to_wait_seq = trigger_end_fpga - last_sample_present;
            time_to_wait_seq += samples_per_data_set;
            float wait_time = time_to_wait_seq * FPGA_PERIOD_NS * 1e-9;
            wait_time = std::min(wait_time, max_wait_time);
            wait_time = std::max(wait_time, min_wait_time);
            // XXX delete dev prints.
            std::cout << "wait for: " << wait_time << std::endl;
            usleep(wait_time * 1e6);
        } else {
            // We have the data we need, break from the loop and copy it out.
            break;
        }
    }

    if (dump_start_frame >= dump_end_frame) {
        // Trigger was too late and missed the data. Return an empty dataset.
        return basebandDumpData(event_id, 0, 0, 0, 0);
    }

    int buf_frame = dump_start_frame % buf->num_frames;
    auto first_meta = (chimeMetadata *) buf->metadata[buf_frame]->metadata;

    stream_id_t stream_id = extract_stream_id(first_meta->stream_ID);
    uint32_t freq_id = bin_number_chime(&stream_id);

    // Figure out how much data we have.
    int64_t data_start_fpga = std::max(trigger_start_fpga, first_meta->fpga_seq_num);
    // For now just assume that we have the last sample, because the locking logic currently waits
    // for it. Could be made to be more robust.
    int64_t data_end_fpga = trigger_end_fpga;

    basebandDumpData dump(
            event_id,
            freq_id,
            num_elements,
            data_start_fpga,
            data_end_fpga - data_start_fpga
            );

    // Fill in the data.
    int64_t next_data_ind = 0;
    for (int frame_index = dump_start_frame; frame_index < dump_end_frame; frame_index++) {
        buf_frame = frame_index % buf->num_frames;
        auto metadata = (chimeMetadata *) buf->metadata[buf_frame]->metadata;
        uint8_t * buf_data = buf->frames[buf_frame];
        int64_t frame_fpga_seq = metadata->fpga_seq_num;
        int64_t frame_ind_start = std::max(data_start_fpga - frame_fpga_seq, (int64_t) 0);
        int64_t frame_ind_end = std::min(data_end_fpga - frame_fpga_seq, (int64_t) samples_per_data_set);
        int64_t data_ind_start = frame_fpga_seq - data_start_fpga + frame_ind_start;
        // The following copy has 0 length unless there is a missing frame.
        nt_memset(
                &dump.data[next_data_ind * num_elements],
                0,
                (data_ind_start - next_data_ind) * num_elements
                );
        // Now copy in the frame data.
        nt_memcpy(
                &dump.data[data_ind_start * num_elements],
                &buf_data[frame_ind_start * num_elements],
                (frame_ind_end - frame_ind_start) * num_elements
                );
        // What data index are we expecting on the next iteration.
        next_data_ind = data_ind_start + frame_ind_end - frame_ind_start;
        // Done with this frame. Allow it to participate in the ring buffer.
        frame_locks[frame_index % num_frames_buffer].unlock();
    }

    unlock_range(dump_start_frame, dump_end_frame);
    return dump;
}

void basebandReadout::lock_range(int start_frame, int end_frame) {
    for (int frame_index = start_frame; frame_index < end_frame; frame_index++) {
        frame_locks[frame_index % num_frames_buffer].lock();
    }
}

void basebandReadout::unlock_range(int start_frame, int end_frame) {
    for (int frame_index = start_frame; frame_index < end_frame; frame_index++) {
        frame_locks[frame_index % num_frames_buffer].unlock();
    }
}



/* Helper for basebandDumpData constructor.
 * Binds a span to an array, aligning it on a 16 byte boundary. Array should
 * be at least 15 bytes too long.
 */
gsl::span<uint8_t> span_from_length_aligned(uint8_t* start, size_t length) {

    intptr_t span_start_int = (intptr_t) start + 15;
    span_start_int -= span_start_int % 16;
    uint8_t* span_start = (uint8_t*) span_start_int;
    uint8_t* span_end = span_start + length;

    return gsl::span<uint8_t>(span_start, span_end);
}


basebandDumpData::basebandDumpData(
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
        // Over allocate so we can align the memory.
        data_ref(new uint8_t[num_elements_ * data_length_fpga_ + 15]),
        data(span_from_length_aligned(data_ref.get(), num_elements_ * data_length_fpga_))
{
}

basebandDumpData::~basebandDumpData() {
}
