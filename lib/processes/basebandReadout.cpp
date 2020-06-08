#include "basebandReadout.hpp"

#include "basebandApiManager.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "gpsTime.h"
#include "nt_memcpy.h"
#include "nt_memset.h"
#include "version.h"
#include "visFileH5.hpp"
#include "visUtil.hpp"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <unistd.h>


using kotekan::basebandApiManager;
using kotekan::basebandDumpStatus;
using kotekan::basebandReadoutManager;
using kotekan::basebandRequest;
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(basebandReadout);


basebandReadout::basebandReadout(Config& config, const string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&basebandReadout::main_thread, this)),
    _base_dir(config.get_default<std::string>(unique_name, "base_dir", "./")),
    _num_frames_buffer(config.get<int>(unique_name, "num_frames_buffer")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    _max_dump_samples(config.get_default<uint64_t>(unique_name, "max_dump_samples", 1 << 30)),
    _write_throttle(config.get_default<float>(unique_name, "write_throttle", 0.)),
    buf(get_buffer("in_buf")),
    next_frame(0),
    oldest_frame(-1),
    frame_locks(_num_frames_buffer),
    // Over allocate so we can align the memory.
    // TODO: Make array of pointers here.
    // TODO: Pointer to array of 8, each of 8 points to a baseband_data
    // TODO: Make this 15 alignment bits PER chunk
    baseband_data(std::make_unique<uint8_t[]>((_num_elements * _max_dump_samples+ 15 )* _num_local_freq))
    {
    // ensure a trailing slash in _base_dir
    if (_base_dir.back() != '/') {
        _base_dir.push_back('/');
    }
    // Get the correlator input meanings, unreordered.
    auto input_reorder = parse_reorder_default(config, unique_name);
    _inputs = std::get<1>(input_reorder);
    auto inputs_copy = _inputs;
    auto order_inds = std::get<0>(input_reorder);
    for (size_t i = 0; i < _inputs.size(); i++) {
        _inputs[order_inds[i]] = inputs_copy[i];
    }

    // Memcopy byte alignments assume the following.
    if (_num_elements % 128) {
        throw std::runtime_error("num_elements must be multiple of 128");
    }

    register_consumer(buf, unique_name.c_str());

    // Ensure input buffer is long enough.
    if (buf->num_frames <= _num_frames_buffer) {
        // This process of creating an error string is rediculous. Figure out what
        // the std::string way to do this is.
        const int msg_len = 200;
        char msg[200];
        snprintf(msg, msg_len, "Input buffer (%d frames) not large enough to buffer %d frames",
                 buf->num_frames, _num_frames_buffer);
        throw std::runtime_error(msg);
    }

    // Make sure output directory is writable.
    if (access(_base_dir.c_str(), W_OK) == 0) {
    } else {
        throw std::runtime_error("Baseband dump directory not writable.");
    }
}

basebandReadout::~basebandReadout() {}

void basebandReadout::main_thread() {

    int frame_id = 0;

    std::unique_ptr<std::thread> wt;
    std::unique_ptr<std::thread> lt;
    basebandReadoutManager *mgrs[_num_local_freq];

    uint32_t freq_ids[_num_local_freq];
    while (!stop_thread) {

        if (wait_for_full_frame(buf, unique_name.c_str(), frame_id % buf->num_frames) == nullptr) {
            break;
        }

        if (!lt) {
            int buf_frame = frame_id % buf->num_frames;
            auto first_meta = (chimeMetadata*)buf->metadata[buf_frame]->metadata;

            stream_id_t stream_id = extract_stream_id(first_meta->stream_ID);
            //basebandReadoutManager<std::reference_wrapper<basebandReadoutManager>> mgrs;
            uint32_t freq_id;
                for(int freqidx=0; freqidx < _num_local_freq;freqidx++){
                    // XXX Map stream IDs to freq IDs with bin_number() (for Pathfinder) or bin_number_chime()
                    // XXX Use num_local_freqs to figure out if we're running on CHIME or PF
                    //if (_num_local_freq == 1){
                    //    freq_id = bin_number_chime(&stream_id);
                    //} else { // more than one freq per stream
                    //    freq_id = bin_number(&stream_id,freqidx);
                    //}
                    freq_id = bin_number_multifreq(&stream_id,_num_local_freq,freqidx);

                    freq_ids[freqidx] = freq_id;
                    mgrs[freqidx] = &(basebandApiManager::instance().register_readout_stage(freq_ids[freqidx]));//TODO: Check register_readout_stage
                    INFO("Starting request-listening thread for freq_id: %" PRIu32, freq_id);

                }
            lt = std::make_unique<std::thread>([&] { this->listen_thread(freq_ids, mgrs); });

            wt = std::make_unique<std::thread>([&] { this->write_thread(mgrs); });
        }

        int done_frame = add_replace_frame(frame_id);
        if (done_frame >= 0) {
            mark_frame_empty(buf, unique_name.c_str(), done_frame % buf->num_frames);
        }

        frame_id++;
    }
    ready_to_write.notify_all();

    if (lt) {
        lt->join();
    }
    if (wt) {
        wt->join();
    }
}

void basebandReadout::listen_thread(const uint32_t freq_ids[], basebandReadoutManager *mgrs[]) {
    //basebandReadoutManager& mgr = *(mgrs[0]);

    std::unique_ptr<basebandReadoutManager::requestStatusMutex> next_requests[_num_local_freq];
    std::shared_ptr<basebandReadoutManager::requestStatusMutex> next_request;
    while (!stop_thread) {
        // Code that listens and waits for triggers and fills in trigger parameters.
        // Latency is *key* here. We want to call get_data within 100ms
        // of L4 sending the trigger.
        next_requests[0] = mgrs[0]->get_next_waiting_request();
        if (next_requests[0]) {
            INFO("Non-null request for freq_ids[idx]: %" PRIu32, freq_ids[0]);
            for(int freqidx = 1; freqidx < _num_local_freq; freqidx++){
                next_requests[freqidx] = mgrs[freqidx]->get_next_waiting_request();
                //INFO("Request for event %" PRIu32,freq_id);
                    //next_requests[freqidx] = next_request;
            INFO("Non-null request for freq_ids[idx]: %" PRIu32, freq_ids[freqidx]);
            }
        

            //loopify
            basebandDumpStatus *basebandDumpStatuses[_num_local_freq];
            std::mutex *request_mtxs[_num_local_freq];
            const basebandRequest *basebandRequests[_num_local_freq];

            for(int freqidx = 0; freqidx < _num_local_freq; freqidx++){
            
                basebandDumpStatuses[freqidx] = &(std::get<0>(*next_requests[freqidx]));
                request_mtxs[freqidx] = &(std::get<1>(*next_requests[freqidx]));

                // This should be safe even without a lock, as there is nothing else
                // yet that can change the dump_status object
                basebandRequest request = basebandDumpStatuses[freqidx]->request;
                basebandRequests[freqidx] = &request;
                // std::time_t tt = std::chrono::system_clock::to_time_t(request.received);
                INFO("Received baseband dump request for event %" PRIu64 ": %" PRIi64
                     " samples starting at count %" PRIi64 ". (next_frame: %d)",
                     request.event_id, request.length_fpga, request.start_fpga, next_frame);
            }//end loopify
            
            //for(int freqidx = 0; freqidx < _num_local_freq; freqidx++){
            //
            //    basebandRequest request = basebandDumpStatuses[freqidx]->request;
            //    // std::time_t tt = std::chrono::system_clock::to_time_t(request.received);
            //    INFO("Received baseband dump request for event %" PRIu64 ": %" PRIi64
            //         " samples starting at count %" PRIi64 ". (next_frame: %d)",
            //         request.event_id, request.length_fpga, request.start_fpga, next_frame);
            //}//end loopify

            //XXX Assume that all eight dumps have the same event ID.
            //const uint64_t event_id = basebandRequests[0]->event_id;
            
            // std::mutex& request_mtx = std::get<1>(*next_requests[0]);
            // basebandDumpStatus& dump_status = std::get<0>(*next_requests[0]);
            // const basebandRequest request = *basebandRequests[0];
            // const uint64_t event_id = request.event_id;
            // INFO("Received baseband dump request for event %" PRIu64 ": %" PRIi64
            //      " samples starting at count %" PRIi64 ". (next_frame: %d)",
            //      event_id, request.length_fpga, request.start_fpga, next_frame);
            
            
            std::unique_lock<std::mutex> lock(dump_to_write_mtx);
            while (dumps_to_write) {
                ready_to_write.wait(lock);
                if (stop_thread)
                    return;
            }
            INFO("Ready to copy samples into the baseband readout buffer");
            
            std::vector<basebandDumpData> dumps_to_write_vec;
            using namespace std::chrono;
            milliseconds ms_before = duration_cast<milliseconds>(
                   system_clock::now().time_since_epoch());
            for(int freqidx = 0; freqidx < _num_local_freq; freqidx++){
                {
                    //loopify
                    std::lock_guard<std::mutex> lock(*request_mtxs[freqidx]);
                    basebandDumpStatuses[freqidx]->state = basebandDumpStatus::State::INPROGRESS;
                    
                    basebandRequest request = basebandDumpStatuses[freqidx]->request;
                    // std::time_t tt = std::chrono::system_clock::to_time_t(request.received);
                    INFO("before get_data(): Received baseband dump request for event %" PRIu64 ": %" PRIi64
                         " samples starting at count %" PRIi64 ". (next_frame: %d)",
                         request.event_id, request.length_fpga, request.start_fpga, next_frame);
                    // Note: the length of the dump still needs to be set with
                    // actual sizes. This is done in `get_data` as it verifies what
                    // is available in the current buffers.
                    // end loopify
                }

                // Copying the data from the ring buffer is done in *this* thread. Writing the data
                // out is done by another thread. This keeps the number of threads that can lock out
                // the main buffer limited to 2 (listen and main).
                // loopify function call,
                //dumps_to_write_vec.push_back(get_data(basebandRequests[freqidx]->event_id, basebandRequests[freqidx]->start_fpga,                             std::min((int64_t)basebandRequests[freqidx]->length_fpga, _max_dump_samples),freqidx));
                dumps_to_write_vec.push_back(get_data((basebandDumpStatuses[freqidx]->request).event_id, 
                                                       (basebandDumpStatuses[freqidx]->request).start_fpga,                             
                                                       std::min((int64_t)(basebandDumpStatuses[freqidx]->request).length_fpga, _max_dump_samples),
                                                       freqidx));
                auto currentData = dumps_to_write_vec.back();
                // At this point we know how much of the requested data we managed to read from the
                // buffer (which may be nothing if the request as recieved too late).
                {
                    std::lock_guard<std::mutex> lock(*request_mtxs[freqidx]);
                    basebandDumpStatuses[freqidx]->bytes_total = currentData.num_elements * currentData.data_length_fpga;
                    basebandDumpStatuses[freqidx]->bytes_remaining = basebandDumpStatuses[freqidx]->bytes_total;
                     if (currentData.data_length_fpga == 0) {
                         INFO("Captured no data for event %" PRIu64 " and freq %" PRIu32 ".", basebandRequests[freqidx]->event_id,
                              freq_ids[freqidx]);
                         basebandDumpStatuses[freqidx]->state = basebandDumpStatus::State::ERROR;
                         basebandDumpStatuses[freqidx]->reason = "No data captured.";
                         continue;
                     } else {
                         INFO("Captured %" PRId64 " samples for event %" PRIu64 " and freq %" PRIu32 ".",
                              currentData.data_length_fpga, currentData.event_id, currentData.freq_id);
                     }
                }
                milliseconds ms_after = duration_cast<milliseconds>(
                        system_clock::now().time_since_epoch());
                INFO("memcpy() call: %d = %d-%d ms", ms_after.count() - ms_before.count(),ms_after.count(),ms_before.count());
                dumps_to_write = std::make_unique<std::vector<basebandDumpData>>(dumps_to_write_vec);//ready to be read out
            }
            lock.unlock();
            ready_to_write.notify_one();
        }
    }
}

void basebandReadout::write_thread(basebandReadoutManager *mgrs[]) {
    while (!stop_thread) {
        std::unique_lock<std::mutex> lock(dump_to_write_mtx);

        // This will reset `dump_to_write` to nullptr so that even if this
        // raises an exception, it will be empty until `listen_thread` copies
        // the next request's data
        while (!dumps_to_write) {
            ready_to_write.wait(lock);
            if (stop_thread)
                return;
        }
        INFO("wt:Dumps to write is not null");

        auto safe_dumps = std::move(dumps_to_write); // nulls dump_to_write, moves data to safe_dumps
        for(int freqidx = 0; freqidx < _num_local_freq; freqidx++){
            auto data = &((*safe_dumps)[freqidx]);//PROGRESS FRONTIER
            auto next_request = mgrs[freqidx]->get_next_ready_request();
            basebandDumpStatus& dump_status = std::get<0>(next_request);
            // Sanity check
            if (dump_status.request.event_id != data->event_id) {
                ERROR("Mismatched event ids: %ld - %ld", dump_status.request.event_id, data->event_id);
                throw std::runtime_error("Mismatched id - abort");
            }
            std::mutex& request_mtx = std::get<1>(next_request);

            try {
                INFO("Before write_dump() for freq_id:%" PRIu32, data->freq_id);
                write_dump(*data, dump_status, request_mtx);
                INFO("After write_dump() for freq_id:%" PRIu32, data->freq_id);
            } catch (HighFive::FileException& e) {
                INFO("Writing Baseband dump file failed with hdf5 error.");
                std::lock_guard<std::mutex> lock(request_mtx);
                dump_status.state = basebandDumpStatus::State::ERROR;
                dump_status.reason = e.what();
            }
        }
        lock.unlock();
        ready_to_write.notify_one();
    }
}

int basebandReadout::add_replace_frame(int frame_id) {
    std::lock_guard<std::mutex> lock(manager_lock);
    int replaced_frame = -1;
    assert(frame_id == next_frame);

    // This will block if we are trying to replace a frame currenty being read out.
    frame_locks[frame_id % _num_frames_buffer].lock();
    // Somehow in C `-1 % _num_frames_buffer == -1` which makes no sence to me.
    // So add `_num_frames_buffer` to `oldest_frame`.
    bool replace_oldest =
        (frame_id % _num_frames_buffer == (oldest_frame + _num_frames_buffer) % _num_frames_buffer);
    if (replace_oldest) {
        replaced_frame = oldest_frame;
        oldest_frame++;
    }
    frame_locks[frame_id % _num_frames_buffer].unlock();

    next_frame++;
    return replaced_frame;
}


basebandDumpData basebandReadout::get_data(uint64_t event_id, int64_t trigger_start_fpga,
                                           int64_t trigger_length_fpga, int freqidx) {
    // This assumes that the frame's timestamps are in order, but not that they
    // are necessarily contiguous.

    int dump_start_frame = 0;
    int dump_end_frame = 0;
    int64_t trigger_end_fpga = trigger_start_fpga + trigger_length_fpga;
    double max_wait_time = 1.;
    double min_wait_time = _samples_per_data_set * FPGA_PERIOD_NS * 1e-9;
    bool advance_info = false;

    if (trigger_length_fpga > _samples_per_data_set * _num_frames_buffer / 2) {
        // Too long, I won't allow it.
        return basebandDumpData();
    }
    while (!stop_thread) {
        int64_t frame_fpga_seq = -1;
        manager_lock.lock();
        dump_start_frame = (oldest_frame > 0) ? oldest_frame : 0;
        dump_end_frame = dump_start_frame;
        //loopify
        for (int frame_index = dump_start_frame; frame_index < next_frame; frame_index++) {
            int buf_frame = frame_index % buf->num_frames;
            auto metadata = (chimeMetadata*)buf->metadata[buf_frame]->metadata;
            frame_fpga_seq = metadata->fpga_seq_num;

            // if the request specified -1 for the start time, use the earliest
            // timestamp available
            if (trigger_start_fpga < 0) {
                trigger_start_fpga = frame_fpga_seq;
                trigger_end_fpga = trigger_start_fpga + trigger_length_fpga;
            }

            if (trigger_end_fpga <= frame_fpga_seq)
                continue;
            if (trigger_start_fpga >= frame_fpga_seq + _samples_per_data_set) {
                dump_start_frame = frame_index + 1;
                continue;
            }
            dump_end_frame = frame_index + 1;
        }
        lock_range(dump_start_frame, dump_end_frame);
        //end loopify

        // Now that the relevant frames are locked, we can unlock the rest of the buffer so
        // it can continue to opperate.
        manager_lock.unlock();

        // Check if the trigger is 'prescient'. That is, if any of the requested data has
        // not yet arrived.
        int64_t last_sample_present = frame_fpga_seq + _samples_per_data_set;
        if (last_sample_present <= trigger_start_fpga + trigger_length_fpga) {
            int64_t time_to_wait_seq = trigger_end_fpga - last_sample_present;
            if (!advance_info) {
                // We only need to print this the first time
                INFO("Advance dump trigger for %" PRIu64 ", waiting for %" PRId64
                     " samples (%.2lf sec)",
                     event_id, time_to_wait_seq, time_to_wait_seq * FPGA_PERIOD_NS / 1e9);
                advance_info = true;
            }
            time_to_wait_seq += _samples_per_data_set;
            double wait_time = time_to_wait_seq * FPGA_PERIOD_NS;
            wait_time = std::min(wait_time, max_wait_time);
            wait_time = std::max(wait_time, min_wait_time);
            std::this_thread::sleep_for(std::chrono::nanoseconds((int)wait_time));
        } else {
            // We have the data we need, break from the loop and copy it out.
            if (advance_info) {
                INFO("Done waiting for dump data for %" PRIu64 ".", event_id);
            }
            break;
        }
        unlock_range(dump_start_frame, dump_end_frame);
    }
    if (stop_thread) {
        return basebandDumpData();
    }

    if (dump_start_frame >= dump_end_frame) {
        // Trigger was too late and missed the data. Return an empty dataset.
        INFO("Baseband dump trigger is too late: %d >= %d", dump_start_frame, dump_end_frame);
        return basebandDumpData();
    }

    int buf_frame = dump_start_frame % buf->num_frames;
    auto first_meta = (chimeMetadata*)buf->metadata[buf_frame]->metadata;

    stream_id_t stream_id = extract_stream_id(first_meta->stream_ID);
    // XXX Map stream IDs to freq IDs with bin_number() (for Pathfinder) or bin_number_chime()
    // XXX Use num_local_freqs to figure out if we're running on CHIME or PF

    uint32_t freq_id = bin_number_multifreq(&stream_id,_num_local_freq,freqidx);
    //if (_num_local_freq == 1){
    //    freq_id = bin_number_chime(&stream_id);
    //} else { // more than one freq per stream
    //    freq_id = bin_number(&stream_id,freqidx);
    //}

    // Figure out how much data we have.
    int64_t data_start_fpga = std::max(trigger_start_fpga, first_meta->fpga_seq_num);
    // For now just assume that we have the last sample, because the locking logic
    // currently waits for it. Could be made to be more robust.
    int64_t data_end_fpga = trigger_end_fpga;

    timeval tmp, delta;
    delta.tv_sec = 0;
    delta.tv_usec = (trigger_start_fpga - first_meta->fpga_seq_num) * FPGA_PERIOD_NS / 1000;
    timeradd(&(first_meta->first_packet_recv_time), &delta, &tmp);
    timespec packet_time0 = {tmp.tv_sec, tmp.tv_usec * 1000};

    //baseband_data.get() is a pointer to preallocated memory. baseband_dataPreallocated at startup
    //Passing baseband_data.get() twice overwrites
    //Allocate 8 times more memory somewhere. Use different memory.

    basebandDumpData dump(event_id, freq_id, _num_elements, data_start_fpga,
                              data_end_fpga - data_start_fpga, packet_time0, baseband_data.get() + freqidx * (_num_elements * _max_dump_samples + 15));
    // Fill in the data.
    int64_t next_data_ind = 0;
    for (int frame_index = dump_start_frame; frame_index < dump_end_frame; frame_index++) {
        buf_frame = frame_index % buf->num_frames;
        auto metadata = (chimeMetadata*)buf->metadata[buf_frame]->metadata;
        uint8_t* buf_data = buf->frames[buf_frame];
        int64_t frame_fpga_seq = metadata->fpga_seq_num;
        int64_t frame_ind_start = std::max(data_start_fpga - frame_fpga_seq, (int64_t)0);
        int64_t frame_ind_end =
            std::min(data_end_fpga - frame_fpga_seq, (int64_t)_samples_per_data_set);
        int64_t data_ind_start = frame_fpga_seq - data_start_fpga + frame_ind_start;
        // The following copy has 0 length unless there is a missing frame.
        nt_memset(&dump.data[next_data_ind * _num_elements], 0,
                  (data_ind_start - next_data_ind) * _num_elements);
        // Now copy in the frame data.
        if (_num_local_freq == 1){
            //using namespace std::chrono;
            //milliseconds ms_before = duration_cast<milliseconds>(
            //        system_clock::now().time_since_epoch());
            nt_memcpy(&dump.data[data_ind_start * _num_elements],
                      &buf_data[frame_ind_start * _num_elements],
                      (frame_ind_end - frame_ind_start) * _num_elements);
            //milliseconds ms_after = duration_cast<milliseconds>(
            //        system_clock::now().time_since_epoch());
            //INFO("memcpy() call: %d = %d-%d ms", ms_after.count() - ms_before.count(),ms_after.count(),ms_before.count());

        } else if (_num_local_freq > 1){
            //using namespace std::chrono;
            //milliseconds ms_before = duration_cast<milliseconds>(
            //        system_clock::now().time_since_epoch());
            for(int timeidx = 0; timeidx < (frame_ind_end - frame_ind_start); timeidx++){
                nt_memcpy(&dump.data[(data_ind_start + timeidx) * _num_elements],
                          &buf_data[((frame_ind_start + timeidx) * _num_local_freq + freqidx) * _num_elements],
                          _num_elements);
            }
            //milliseconds ms_after = duration_cast<milliseconds>(
            //      system_clock::now().time_since_epoch());
            //INFO("memcpy() call: %d = %d-%d ms", ms_after.count() - ms_before.count(),ms_after.count(),ms_before.count());
        }
        // What data index are we expecting on the next iteration.
        next_data_ind = data_ind_start + frame_ind_end - frame_ind_start;
        // Done with this frame. Allow it to participate in the ring buffer.
        frame_locks[frame_index % _num_frames_buffer].unlock();
    }
    unlock_range(dump_start_frame, dump_end_frame);

    if (stop_thread)
        return basebandDumpData();
    return dump;
}

void basebandReadout::lock_range(int start_frame, int end_frame) {
    for (int frame_index = start_frame; frame_index < end_frame; frame_index++) {
        frame_locks[frame_index % _num_frames_buffer].lock();
    }
}

void basebandReadout::unlock_range(int start_frame, int end_frame) {
    for (int frame_index = start_frame; frame_index < end_frame; frame_index++) {
        frame_locks[frame_index % _num_frames_buffer].unlock();
    }
}

void basebandReadout::write_dump(basebandDumpData data, basebandDumpStatus& dump_status,
                                 std::mutex& request_mtx) {

    // TODO Create parent directories.
    std::string filename =
        _base_dir + dump_status.request.file_path + "/" + dump_status.request.file_name;
    std::string lock_filename = create_lockfile(filename);
    INFO(("Writing baseband dump to " + filename).c_str());

    auto file = HighFive::File(filename, HighFive::File::ReadWrite | HighFive::File::Create
                                             | HighFive::File::Truncate);

    std::string version = "NT_3.1.0";
    file.createAttribute<std::string>("archive_version", HighFive::DataSpace::From(version))
        .write(version);

    std::string git_version = get_git_commit_hash();
    file.createAttribute<std::string>("git_version_tag", HighFive::DataSpace::From(git_version))
        .write(git_version);

    char temp[256];
    std::string username = (getlogin_r(temp, 256) == 0) ? temp : "unknown";
    file.createAttribute<std::string>("system_user", HighFive::DataSpace::From(username))
        .write(username);

    gethostname(temp, 256);
    std::string hostname = temp;
    file.createAttribute<std::string>("collection_server", HighFive::DataSpace::From(hostname))
        .write(hostname);

    file.createAttribute<uint64_t>("event_id", HighFive::DataSpace::From(data.event_id))
        .write(data.event_id);

    file.createAttribute<uint32_t>("freq_id", HighFive::DataSpace::From(data.freq_id))
        .write(data.freq_id);

    double freq = freq_from_bin(data.freq_id);
    file.createAttribute<double>("freq", HighFive::DataSpace::From(freq)).write(freq);

    file.createAttribute<uint64_t>("time0_fpga_count",
                                   HighFive::DataSpace::From(data.data_start_fpga))
        .write(data.data_start_fpga);

    double ptime = ts_to_double(data.data_start_ctime);
    file.createAttribute<double>("first_packet_recv_time", HighFive::DataSpace::From(ptime))
        .write(ptime);

    timespec time0;
    std::string time0_type;
    if (is_gps_global_time_set()) {
        time0 = compute_gps_time(data.data_start_fpga);
        time0_type = "GPS";
    } else {
        time0 = data.data_start_ctime;
        time0_type = "PACKET_RECV";
    }
    double ftime0 = ts_to_double(time0);
    double ftime0_offset = (time0.tv_nsec - fmod(ftime0, 1.) * 1e9) / 1e9;
    file.createAttribute<double>("time0_ctime", HighFive::DataSpace::From(ftime0)).write(ftime0);
    file.createAttribute<double>("time0_ctime_offset", HighFive::DataSpace::From(ftime0_offset))
        .write(ftime0_offset);
    file.createAttribute<std::string>("type_time0_ctime", HighFive::DataSpace::From(time0_type))
        .write(time0_type);
    double delta_t = (double)FPGA_PERIOD_NS / 1e9;
    file.createAttribute<double>("delta_time", HighFive::DataSpace::From(delta_t)).write(delta_t);

    size_t num_elements = data.num_elements;
    size_t ntime_chunk = TARGET_CHUNK_SIZE / num_elements;

    std::vector<size_t> cur_dims = {0, num_elements};
    std::vector<size_t> max_dims = {data.data_length_fpga, num_elements};
    std::vector<size_t> chunk_dims = {ntime_chunk, num_elements};

    auto index_map = file.createGroup("index_map");
    index_map
        .createDataSet("input", HighFive::DataSpace::From(_inputs),
                       HighFive::create_datatype<input_ctype>())
        .write(_inputs);

    auto space = HighFive::DataSpace(cur_dims, max_dims);
    // HighFive::DataSetCreateProps props;
    // props.add(HighFive::Chunking(chunk_dims));
    auto dataset =
        file.createDataSet("baseband", space, HighFive::create_datatype<uint8_t>(), chunk_dims);

    std::vector<std::string> axes = {"time", "input"};
    dataset.createAttribute<std::string>("axis", HighFive::DataSpace::From(axes)).write(axes);

    size_t ii_samp = 0;
    while (!stop_thread) {
        size_t to_write = std::min((size_t)data.data_length_fpga - ii_samp, ntime_chunk);
        dataset.resize({ii_samp + to_write, num_elements});
        dataset.select({ii_samp, 0, 0}, {to_write, num_elements})
            .write((uint8_t**)&(data.data[ii_samp * num_elements]));
        file.flush();

        {
            std::lock_guard<std::mutex> lock(request_mtx);
            dump_status.bytes_remaining -= to_write * num_elements;
        }
        ii_samp += ntime_chunk;
        if (ii_samp >= data.data_length_fpga)
            break;
        // Add intentional throttling.
        float stime = _write_throttle * to_write * FPGA_PERIOD_NS;
        std::this_thread::sleep_for(std::chrono::nanoseconds((int)stime));
    }
    std::remove(lock_filename.c_str());

    if (ii_samp > data.data_length_fpga) {
        std::lock_guard<std::mutex> lock(request_mtx);
        dump_status.state = basebandDumpStatus::State::DONE;
        INFO("Baseband dump for event %" PRIu64 ", freq %" PRIu32 " complete.", data.event_id,
             data.freq_id);
    } else {
        std::lock_guard<std::mutex> lock(request_mtx);
        dump_status.state = basebandDumpStatus::State::ERROR;
        dump_status.reason = "Kotekan exit before write complete.";
        INFO("Baseband dump for event %" PRIu64 ", freq %" PRIu32 " incomplete.", data.event_id,
             data.freq_id);
    }
    // H5 file goes out of scope and is closed automatically.
}


/* Helper for basebandDumpData constructor.
 * Binds a span to an array, aligning it on a 16 byte boundary. Array should
 * be at least 15 bytes too long.
 */
gsl::span<uint8_t> span_from_length_aligned(uint8_t* start, size_t length) {

    intptr_t span_start_int = (intptr_t)start + 15;
    span_start_int -= span_start_int % 16;
    uint8_t* span_start = (uint8_t*)span_start_int;
    uint8_t* span_end = span_start + length;

    return gsl::span<uint8_t>(span_start, span_end);
}


basebandDumpData::basebandDumpData(uint64_t event_id_, uint32_t freq_id_, uint32_t num_elements_,
                                   int64_t data_start_fpga_, uint64_t data_length_fpga_,
                                   timespec data_start_ctime_, uint8_t* baseband_data) :
    event_id(event_id_),
    freq_id(freq_id_),
    num_elements(num_elements_),
    data_start_fpga(data_start_fpga_),
    data_length_fpga(data_length_fpga_),
    data_start_ctime(data_start_ctime_),
    data(span_from_length_aligned(baseband_data, num_elements_ * data_length_fpga_)) {}

basebandDumpData::basebandDumpData() : basebandDumpData(0, 0, 0, 0, 0, {0, 0}, nullptr) {}
