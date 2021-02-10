#include "basebandReadout.hpp"

#include "BasebandMetadata.hpp"   // for BasebandMetadata
#include "Config.hpp"             // for Config
#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"          // for Telescope
#include "basebandApiManager.hpp" // for basebandApiManager
#include "buffer.h"               // for Buffer, mark_frame_empty, register_consumer, wait_fo...
#include "bufferContainer.hpp"    // for bufferContainer
#include "chimeMetadata.hpp"      // for chimeMetadata
#include "kotekanLogging.hpp"     // for INFO, DEBUG, ERROR
#include "metadata.h"             // for metadataContainer
#include "nt_memcpy.h"            // for nt_memcpy
#include "nt_memset.h"            // for nt_memset
#include "prometheusMetrics.hpp"  // for Counter, Gauge, MetricFamily, Metrics
#include "version.h"              // for get_git_commit_hash
#include "visFile.hpp"            // for create_lockfile
#include "visFileH5.hpp"          // for create_datatype
#include "visUtil.hpp"            // for input_ctype, ts_to_double, parse_reorder_default

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span, operator!=

#include <algorithm>                // for max, copy, copy_backward, min
#include <assert.h>                 // for assert
#include <atomic>                   // for atomic_bool
#include <chrono>                   // for system_clock::time_point, system_clock, nanoseconds
#include <cstdint>                  // for uint64_t, uint8_t
#include <cstdio>                   // for remove, snprintf
#include <deque>                    // for deque
#include <exception>                // for exception
#include <functional>               // for _Bind_helper<>::type, bind, function
#include <highfive/H5Attribute.hpp> // for Attribute, Attribute::write
#include <highfive/H5DataSet.hpp>   // for AnnotateTraits::createAttribute, DataSet, DataSet::r...
#include <highfive/H5DataSpace.hpp> // for DataSpace::From, DataSpace, DataSpace::DataSpace
#include <highfive/H5DataType.hpp>  // for create_datatype
#include <highfive/H5Exception.hpp> // for FileException
#include <highfive/H5File.hpp>      // for File, NodeTraits::createDataSet, File::Create, File:...
#include <highfive/H5Group.hpp>     // for Group
#include <highfive/H5Selection.hpp> // for Selection, SliceTraits::write, SliceTraits::select
#include <math.h>                   // for fmod
#include <memory>                   // for unique_ptr, make_shared, make_unique, allocator_trai...
#include <regex>                    // for match_results<>::_Base_type
#include <stdexcept>                // for runtime_error
#include <sys/time.h>               // for timeval, timeradd
#include <thread>                   // for thread, sleep_for
#include <time.h>                   // for timespec
#include <tuple>                    // for get
#include <unistd.h>                 // for access, gethostname, getlogin_r, W_OK
#include <utility>                  // for get


using kotekan::basebandApiManager;
using kotekan::basebandDumpData;
using kotekan::basebandDumpStatus;
using kotekan::basebandReadoutManager;
using kotekan::basebandRequest;
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(basebandReadout);


basebandReadout::basebandReadout(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&basebandReadout::main_thread, this)),
    _num_frames_buffer(config.get<int>(unique_name, "num_frames_buffer")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    _max_dump_samples(config.get_default<uint64_t>(unique_name, "max_dump_samples", 1 << 30)),
    in_buf(get_buffer("in_buf")),
    next_frame(0),
    oldest_frame(-1),
    frame_locks(_num_frames_buffer),
    out_buf(get_buffer("out_buf")),
    out_frame_id(out_buf),
    readout_counter(kotekan::prometheus::Metrics::instance().add_counter(
        "kotekan_baseband_readout_total", unique_name, {"freq_id", "status"})),
    readout_in_progress_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_baseband_readout_in_progress", unique_name, {"freq_id"})) {

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

    register_consumer(in_buf, unique_name.c_str());

    register_producer(out_buf, unique_name.c_str());

    // Ensure input buffer is long enough.
    if (in_buf->num_frames <= _num_frames_buffer) {
        // This process of creating an error std::string is rediculous. Figure out what
        // the std::string way to do this is.
        const int msg_len = 200;
        char msg[200];
        snprintf(msg, msg_len, "Input buffer (%d frames) not large enough to buffer %d frames",
                 in_buf->num_frames, _num_frames_buffer);
        throw std::runtime_error(msg);
    }
}

void basebandReadout::main_thread() {

    auto& tel = Telescope::instance();
    int frame_id = 0;

    std::unique_ptr<std::thread> lt;

    basebandReadoutManager* mgr = nullptr;
    while (!stop_thread) {

        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id % in_buf->num_frames)
            == nullptr) {
            break;
        }

        if (!lt) {
            int in_buf_frame = frame_id % in_buf->num_frames;
            uint32_t freq_id = tel.to_freq_id(in_buf, in_buf_frame);

            DEBUG("Initialize baseband metrics for freq_id: {:d}", freq_id);
            readout_counter.labels({std::to_string(freq_id), "done"});
            readout_counter.labels({std::to_string(freq_id), "error"});
            readout_counter.labels({std::to_string(freq_id), "no_data"});
            readout_in_progress_metric.labels({std::to_string(freq_id)}).set(0);

            INFO("Starting request-listening thread for freq_id: {:d}", freq_id);
            mgr = &basebandApiManager::instance().register_readout_stage(freq_id);
            lt = std::make_unique<std::thread>([&] { this->readout_thread(freq_id, *mgr); });
        }

        int done_frame = add_replace_frame(frame_id);
        if (done_frame >= 0) {
            mark_frame_empty(in_buf, unique_name.c_str(), done_frame % in_buf->num_frames);
        }

        frame_id++;
    }
    if (mgr) {
        mgr->stop();
    }

    if (lt) {
        lt->join();
    }
}

void basebandReadout::readout_thread(const uint32_t freq_id, basebandReadoutManager& mgr) {
    while (!stop_thread) {
        // Code that listens and waits for triggers and fills in trigger parameters.
        // Latency is *key* here. We want to call extract_data within 100ms
        // of L4 sending the trigger.

        auto next_request = mgr.get_next_waiting_request();

        if (next_request) {
            basebandDumpStatus& dump_status = std::get<0>(*next_request);
            std::mutex& request_mtx = std::get<1>(*next_request);

            start_processing(dump_status, request_mtx);

            const basebandRequest request = dump_status.request;
            auto data = wait_for_data(request.event_id, freq_id, request.start_fpga,
                                      std::min((int64_t)request.length_fpga, _max_dump_samples));
            basebandDumpData::Status status = data.status;

            if (status == basebandDumpData::Status::Ok) {
                status = extract_data(data);
            }

            end_processing(status, freq_id, dump_status, request_mtx);
        }
    }
}

void basebandReadout::start_processing(basebandDumpStatus& dump_status, std::mutex& request_mtx) {
    // Reading the request parameters should be safe even without a
    // lock, as they are read-only once received.
    const basebandRequest request = dump_status.request;
    INFO("Received baseband dump request for event {:d}: {:d} samples starting at count "
         "{:d}. (next_frame: {:d})",
         request.event_id, request.length_fpga, request.start_fpga, next_frame);

    {
        std::lock_guard<std::mutex> lock(request_mtx);
        dump_status.state = basebandDumpStatus::State::INPROGRESS;
        dump_status.started = std::make_shared<std::chrono::system_clock::time_point>(
            std::chrono::system_clock::now());
        // Note: the length of the dump still needs to be set with
        // actual sizes. This is done in `extract_data` as it verifies what
        // is available in the current buffers.
    }
}

void basebandReadout::end_processing(basebandDumpData::Status status, const uint32_t freq_id,
                                     basebandDumpStatus& dump_status, std::mutex& request_mtx) {
    auto& request_no_data_counter = readout_counter.labels({std::to_string(freq_id), "no_data"});

    // At this point we know how much of the requested data we managed to read from the
    // buffer (which may be nothing if the request as received too late).
    {
        std::lock_guard<std::mutex> lock(request_mtx);
        if (status != basebandDumpData::Status::Ok) {
            INFO("Captured no data for event {:d} and freq {:d}.", dump_status.request.event_id,
                 freq_id);
            dump_status.state = basebandDumpStatus::State::ERROR;
            dump_status.finished = std::make_shared<std::chrono::system_clock::time_point>(
                std::chrono::system_clock::now());
            switch (status) {
                case basebandDumpData::Status::TooLong:
                    dump_status.reason = "Request length exceeds the configured limit.";
                    break;
                case basebandDumpData::Status::Late:
                    dump_status.reason = "No data captured.";
                    request_no_data_counter.inc();
                    break;
                case basebandDumpData::Status::ReserveFailed:
                    dump_status.reason = "No free space in the baseband buffer";
                    break;
                case basebandDumpData::Status::Cancelled:
                    dump_status.reason = "Kotekan exiting.";
                    break;
                default:
                    INFO("Unknown dump status: {}", int(status));
                    throw std::runtime_error(
                        "Unhandled basebandDumpData::Status case in a switch statement.");
            }
        } else {
            dump_status.state = basebandDumpStatus::State::DONE;
        }
        dump_status.finished = std::make_shared<std::chrono::system_clock::time_point>(
            std::chrono::system_clock::now());
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

basebandDumpData basebandReadout::wait_for_data(const uint64_t event_id, const uint32_t freq_id,
                                                int64_t trigger_start_fpga,
                                                int64_t trigger_length_fpga) {
    DEBUG("Waiting for samples to copy into the baseband readout buffer");

    if (trigger_length_fpga > _samples_per_data_set * _num_frames_buffer / 2) {
        // Too long, I won't allow it.
        return basebandDumpData::Status::TooLong;
    }

    // This assumes that the frame's timestamps are in order, but not that they
    // are necessarily contiguous.
    auto& tel = Telescope::instance();
    const double fpga_period_s = ts_to_double(tel.seq_length());

    int dump_start_frame = 0;
    int dump_end_frame = 0;
    int64_t trigger_end_fpga = trigger_start_fpga + trigger_length_fpga;
    double max_wait_time = 1.;
    double min_wait_time = _samples_per_data_set * fpga_period_s;
    bool advance_info = false;

    while (!stop_thread) {
        int64_t frame_fpga_seq = -1;
        manager_lock.lock();
        dump_start_frame = (oldest_frame > 0) ? oldest_frame : 0;
        dump_end_frame = dump_start_frame;

        for (int frame_index = dump_start_frame; frame_index < next_frame; frame_index++) {
            int in_buf_frame = frame_index % in_buf->num_frames;
            auto metadata = (chimeMetadata*)in_buf->metadata[in_buf_frame]->metadata;
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

        // Now that the relevant frames are locked, we can unlock the rest of the buffer so
        // it can continue to operate.
        manager_lock.unlock();

        // Check if the trigger is 'prescient'. That is, if any of the requested data has
        // not yet arrived.
        int64_t last_sample_present = frame_fpga_seq + _samples_per_data_set;
        if (last_sample_present <= trigger_start_fpga + trigger_length_fpga) {
            int64_t time_to_wait_seq = trigger_end_fpga - last_sample_present;
            if (!advance_info) {
                // We only need to print this the first time
                INFO("Advance dump trigger for {:d}, waiting for {:d} samples ({:.2f} sec)",
                     event_id, time_to_wait_seq, time_to_wait_seq * fpga_period_s);
                advance_info = true;
            }
            time_to_wait_seq += _samples_per_data_set;
            double wait_time = time_to_wait_seq * fpga_period_s * 1e9;
            wait_time = std::min(wait_time, max_wait_time);
            wait_time = std::max(wait_time, min_wait_time);
            std::this_thread::sleep_for(std::chrono::nanoseconds((int)wait_time));
        } else {
            // We have the data we need, break from the loop and copy it out.
            if (advance_info) {
                INFO("Done waiting for dump data for {:d}.", event_id);
            }
            break;
        }
        unlock_range(dump_start_frame, dump_end_frame);
    }
    if (stop_thread) {
        return basebandDumpData::Status::Cancelled;
    } else if (dump_start_frame >= dump_end_frame) {
        // Trigger was too late and missed the data. Return an empty dataset.
        INFO("Baseband dump trigger is too late: {:d} >= {:d}", dump_start_frame, dump_end_frame);
        return basebandDumpData::Status::Late;
    } else {
        INFO("Dump data ready for {:d}/{:d}: frames {:d}-{:d}.", event_id, freq_id,
             dump_start_frame, dump_end_frame);
        return basebandDumpData(event_id, freq_id, trigger_start_fpga, trigger_length_fpga,
                                dump_start_frame, dump_end_frame);
    }
}

basebandDumpData::Status basebandReadout::extract_data(basebandDumpData data) {
    DEBUG("Ready to copy samples into the baseband readout buffer");
    assert(data.dump_start_frame < data.dump_end_frame);

    const uint64_t event_id = data.event_id;

    int in_buf_frame = data.dump_start_frame % in_buf->num_frames;
    auto first_meta = (chimeMetadata*)in_buf->metadata[in_buf_frame]->metadata;

    const uint32_t freq_id = data.freq_id;

    // Figure out how much data we have.
    int64_t data_start_fpga = std::max(data.trigger_start_fpga, first_meta->fpga_seq_num);
    // For now just assume that we have the last sample, because the locking logic
    // currently waits for it. Could be made to be more robust.
    int64_t data_end_fpga = data.trigger_start_fpga + data.trigger_length_fpga;

    INFO("Dump data for {:d}/{:d}: frames {:d}-{:d}; samples {}-{}.", event_id, freq_id,
         data.dump_start_frame, data.dump_end_frame, data_start_fpga, data_end_fpga);

    std::vector<uint8_t> empty;
    basebandDumpData dump(basebandDumpData::Status::Cancelled);
    // Fill in the data.
    int64_t next_data_ind = 0;
    for (int frame_index = data.dump_start_frame; !stop_thread && frame_index < data.dump_end_frame;
         frame_index++) {
        in_buf_frame = frame_index % in_buf->num_frames;
        auto metadata = (chimeMetadata*)in_buf->metadata[in_buf_frame]->metadata;
        uint8_t* in_buf_data = in_buf->frames[in_buf_frame];
        int64_t frame_fpga_seq = metadata->fpga_seq_num;
        int64_t frame_ind_start = std::max(data_start_fpga - frame_fpga_seq, (int64_t)0);
        int64_t frame_ind_end =
            std::min(data_end_fpga - frame_fpga_seq, (int64_t)_samples_per_data_set);
        int64_t data_ind_start = frame_fpga_seq - data_start_fpga + frame_ind_start;
        // The following copy has 0 length unless there is a missing frame.
        INFO("Copy samples {}/{}-{} to {} ({})", frame_index, frame_ind_start, frame_ind_end,
             out_frame_id, next_data_ind);

        // check if there is space for it in the output buffer
        if (is_frame_empty(out_buf, out_frame_id)) {
            // This call cannot block because of the check above.
            uint8_t* frame_out = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
            // if (frame_out == nullptr)
            //     break;
            if (frame_out != nullptr) {
                memcpy(in_buf_data + frame_ind_start,
                       out_buf->frames[out_frame_id],
                       _num_elements * (frame_ind_end - frame_ind_start));

                allocate_new_metadata_object(out_buf, out_frame_id);
                BasebandMetadata* out_metadata =
                    (BasebandMetadata*)get_metadata(out_buf, out_frame_id);

                out_metadata->event_id = event_id;
                out_metadata->freq_id = freq_id;
                out_metadata->start = data.trigger_start_fpga;
                out_metadata->end = data.trigger_length_fpga;
                out_metadata->fpga_seq = frame_fpga_seq + frame_ind_start;
                out_metadata->valid_from = 0;
                out_metadata->valid_to = frame_ind_end - frame_ind_start;

                mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);
            }
        } else {
            WARN("Output buffer full ({:d}). Dropping frame {:d}/{:d}", out_frame_id, event_id,
                 next_data_ind);
        }

        // What data index are we expecting on the next iteration.
        next_data_ind = data_ind_start + frame_ind_end - frame_ind_start;
        // Done with this frame. Allow it to participate in the ring buffer.
        frame_locks[frame_index % _num_frames_buffer].unlock();
    }
    unlock_range(data.dump_start_frame, data.dump_end_frame);

    if (stop_thread) {
        return basebandDumpData::Status::Cancelled;
    } else {
        return basebandDumpData::Status::Ok;
    }
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
