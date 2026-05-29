#include "basebandReadout.hpp"

#include "BasebandMetadata.hpp"   // for BasebandMetadata
#include "Config.hpp"             // for Config
#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"          // for Telescope
#include "basebandApiManager.hpp" // for basebandApiManager
#include "buffer.hpp"             // for Buffer
#include "chordMetadata.hpp"             // for chordMetadata
#include "kotekanLogging.hpp"    // for INFO, DEBUG, WARN
#include "prometheusMetrics.hpp" // for Counter, Gauge, MetricFamily, Metrics
#include "visUtil.hpp"           // for input_ctype, frameID, ts_to_double, modulo, parse_reor...

#include "fmt.hpp" // for compile_string_to_view, join

#include <algorithm>     // for max, copy, equal, min
#include <assert.h>      // for assert
#include <bits/chrono.h> // for system_clock, nanoseconds
#include <cstdint>       // for int64_t, uint32_t, uint64_t, uint8_t
#include <cstdio>        // for snprintf
#include <ctime>         // for timespec
#include <functional>    // for bind, function
#include <math.h>        // for fmod
#include <memory>        // for shared_ptr, unique_ptr, make_shared, make_unique
#include <stdexcept>     // for runtime_error
#include <string.h>      // for memcpy, memset
#include <sys/time.h>    // for timeval, timeradd
#include <thread>        // for thread, sleep_for
#include <tuple>         // for get
#include <utility>       // for pair


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
    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_per_stream(config.get_default<uint32_t>(unique_name, "_num_freq_per_stream", 1)),
    _samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    _max_dump_samples(config.get_default<uint64_t>(unique_name, "max_dump_samples", 1 << 30)),
    _num_beams(config.get_default<uint32_t>(unique_name, "num_beams", 1)),
    _datasets_per_scan(config.get_default<int>(unique_name, "datasets_per_scan", 1)),
    _datasets_per_delay_update(config.get_default<int>(unique_name, "datasets_per_delay_update", 0)),
    in_buf(get_buffer("in_buf")), next_frame(0), oldest_frame(-1), frame_locks(_num_frames_buffer),
    out_buf(get_buffer("out_buf")), out_frame_id(out_buf),
    outmb_buf(get_buffer("outmb_buf")), outmb_frame_id(outmb_buf),
    readout_counter(kotekan::prometheus::Metrics::instance().add_counter(
        "kotekan_baseband_readout_total", unique_name, {"freq_id", "status"})),
    readout_sent_frame_counter(kotekan::prometheus::Metrics::instance().add_counter(
        "kotekan_baseband_readout_sent_frames_total", unique_name, {"freq_id"})),
    readout_dropped_frame_counter(kotekan::prometheus::Metrics::instance().add_counter(
        "kotekan_baseband_readout_dropped_frames_total", unique_name, {"freq_id"})),
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
    INFO("finished constructor, num_beams: {}", _num_beams);
    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
    outmb_buf->register_producer(unique_name);
    
    // Try to get gain_tracking_buffer if doing multibeam beamforming
    if (_num_beams > 0) {
        try {
            _gain_tracking_buffer = get_buffer("gain_tracking_buffer");
            INFO("Loaded gain_tracking_buffer for multibeam beamforming with {} beams", _num_beams);
        } catch (const std::exception& e) {
            WARN("Could not load gain_tracking_buffer for multibeam beamforming: {}", e.what());
            _gain_tracking_buffer = nullptr;
        }
    }

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
    int frame_id = 0;

    std::unique_ptr<std::thread> lt;

    std::vector<basebandReadoutManager*> mgrs;
    uint32_t freq_ids[_num_freq_per_stream];
    while (!stop_thread) {

        if (in_buf->wait_for_full_frame(unique_name, frame_id % in_buf->num_frames) == nullptr) {
            break;
        }

        if (!lt) {
            const auto fpga0_tv = tel.to_time(0);
            fpga0_ns = fpga0_tv.tv_sec * 1'000'000'000 + fpga0_tv.tv_nsec;
            int in_buf_frame = frame_id % in_buf->num_frames;
            for (uint32_t stream_freq_idx = 0; stream_freq_idx < _num_freq_per_stream;
                 ++stream_freq_idx) {
                uint32_t freq_id =
                    get_chord_metadata(in_buf, in_buf_frame)->get_coarse_freq()[stream_freq_idx];
                freq_ids[stream_freq_idx] = freq_id;
                INFO("Configuring baseband readout for freq_id: {} (stream_freq_idx: {})", freq_id, stream_freq_idx);

                DEBUG("Initialize baseband metrics for freq_id: {:d}/{:d}", freq_id,
                      stream_freq_idx);
                readout_counter.labels({std::to_string(freq_id), "done"});
                readout_counter.labels({std::to_string(freq_id), "error"});
                readout_counter.labels({std::to_string(freq_id), "no_data"});
                readout_sent_frame_counter.labels({std::to_string(freq_id)});
                readout_dropped_frame_counter.labels({std::to_string(freq_id)});
                readout_in_progress_metric.labels({std::to_string(freq_id)}).set(0);

                basebandReadoutManager* mgr =
                    &basebandApiManager::instance().register_readout_stage(freq_id);
                mgrs.push_back(mgr);
            }
            INFO("Starting request-listening thread for freq_id: {}",
                 fmt::join(freq_ids, freq_ids + _num_freq_per_stream, ", "));
            lt = std::make_unique<std::thread>([&] { this->readout_thread(freq_ids, mgrs); });
        }

        int done_frame = add_replace_frame(frame_id);
        if (done_frame >= 0) {
            in_buf->mark_frame_empty(unique_name, done_frame % in_buf->num_frames);
        }

        frame_id++;
    }
    for (auto mgr : mgrs) {
        mgr->stop();
    }

    if (lt) {
        lt->join();
    }
}

void basebandReadout::readout_thread(const uint32_t freq_ids[],
                                     const std::vector<basebandReadoutManager*>& mgrs) {
    while (!stop_thread) {
        // Code that listens and waits for triggers and fills in trigger parameters.
        // Latency is *key* here. We want to call extract_data within 100ms
        // of L4 sending the trigger.

        if (auto next_request = mgrs[0]->get_next_waiting_request()) {
            for (uint32_t stream_freq_idx = 0; stream_freq_idx < _num_freq_per_stream;
                 ++stream_freq_idx) {
                uint32_t freq_id = freq_ids[stream_freq_idx];

                // the first frequency's request was retrieved as part of the
                // top if-statement, but the rest still need to be done
                if (stream_freq_idx) {
                    next_request = mgrs[stream_freq_idx]->get_next_waiting_request();
                }

                // basebandDumpStatus& dump_status, std::mutex& request_mtx
                auto [dump_status, request_mtx] = *next_request;

                start_processing(dump_status, request_mtx);
                
                readout_in_progress_metric.labels({std::to_string(freq_id)}).set(1);
                const basebandRequest request = dump_status.request;
                INFO("Entering wait_for_data");
                auto data =
                    wait_for_data(request.event_id, freq_id, stream_freq_idx, request.start_fpga,
                                  std::min((int64_t)request.length_fpga, _max_dump_samples));
                basebandDumpData::Status status = data.status;

                if (status == basebandDumpData::Status::Ok) {
                    status = extract_data(data);
                }
                readout_in_progress_metric.labels({std::to_string(freq_id)}).set(0);

                end_processing(status, freq_id, dump_status, request_mtx);
            }
        }
    }
}

void basebandReadout::start_processing(basebandDumpStatus& dump_status, std::mutex& request_mtx) {
    // Reading the request parameters should be safe even without a
    // lock, as they are read-only once received.
    const basebandRequest request = dump_status.request;
    
    timespec fpga_start_ts = tel.to_time(0);
    INFO("start_processing: FPGA frame 0 corresponds to time {}.{} ({} ns)", fpga_start_ts.tv_sec, fpga_start_ts.tv_nsec, fpga_start_ts.tv_nsec);
    timespec trigger_start_ts = tel.to_time(request.start_fpga);
    double ftime0 = ts_to_double(trigger_start_ts);
    INFO("Received baseband dump request for event {:d}: {:d} baseband samples starting at fpga_count "
         "{:d} corresponding to unix time={:.9f}) (next_frame: {:d})",
         request.event_id, request.length_fpga, request.start_fpga, ftime0, next_frame); // TODO: Rename request.start_fpga to request.dump_start_fpga

    {
        std::lock_guard<std::mutex> lock(request_mtx);
        dump_status.state = basebandDumpStatus::State::INPROGRESS;
        dump_status.started = std::make_shared<std::chrono::system_clock::time_point>(
            std::chrono::system_clock::now());
        // Note: the length of the dump still needs to be set with
        // actual sizes. This is done in `extract_data` as it verifies what
        // is available in the current buffers.
    }
    INFO("Exiting start_processing");
}

void basebandReadout::end_processing(basebandDumpData::Status status, const uint32_t freq_id,
                                     basebandDumpStatus& dump_status, std::mutex& request_mtx) {
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
                    readout_counter.labels({std::to_string(freq_id), "error"}).inc();
                    break;
                case basebandDumpData::Status::Late:
                    dump_status.reason = "No data captured.";
                    readout_counter.labels({std::to_string(freq_id), "no_data"}).inc();
                    break;
                case basebandDumpData::Status::ReserveFailed:
                    dump_status.reason = "No free space in the baseband buffer";
                    readout_counter.labels({std::to_string(freq_id), "error"}).inc();
                    break;
                case basebandDumpData::Status::Cancelled:
                    dump_status.reason = "Kotekan exiting.";
                    readout_counter.labels({std::to_string(freq_id), "error"}).inc();
                    break;
                default:
                    INFO("Unknown dump status: {}", int(status));
                    throw std::runtime_error(
                        "Unhandled basebandDumpData::Status case in a switch statement.");
            }
        } else {
            dump_status.state = basebandDumpStatus::State::DONE;
            readout_counter.labels({std::to_string(freq_id), "done"}).inc();
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
                                                const uint32_t stream_freq_idx,
                                                int64_t trigger_start_fpga,
                                                int64_t trigger_length_fpga) {
    DEBUG("Waiting for samples to copy into the baseband readout buffer");

    if (trigger_length_fpga > _samples_per_data_set * _num_frames_buffer / 2) {
        // Too long, I won't allow it.
        return basebandDumpData::Status::TooLong;
    }

    // This assumes that the frame's timestamps are in order, but not that they
    // are necessarily contiguous.
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
            INFO("wait_for_data: frame_index, dump_start_frame, dump_end_frame: {}, {}, {}", frame_index, dump_start_frame, dump_end_frame);
            int in_buf_frame = frame_index % in_buf->num_frames;
            auto metadata = get_chord_metadata(in_buf, in_buf_frame);
            frame_fpga_seq = metadata->get_fpga_seq_num();

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
	int lock_start = std::max(oldest_frame, dump_start_frame - _datasets_per_scan);
	int lock_end = dump_end_frame + 2 * _datasets_per_scan;
    DEBUG("Ready to lock frames. Finding {} calibrators using start of lock_range = {} - {}", _num_beams, lock_start, lock_end);
        lock_range(lock_start, lock_end);
	DEBUG("Frames locked. Finding {} calibrators using start of lock_range = {} - {}", _num_beams, lock_start, lock_end);

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
        return basebandDumpData(event_id, freq_id, stream_freq_idx, trigger_start_fpga,
                                trigger_length_fpga, dump_start_frame, dump_end_frame);
    }
}

// Compute beamforming phases for multiple beams
void basebandReadout::compute_beam_phases(Config& config, const std::string& unique_name, 
                                          float* phases, time_t beamform_time,
                                          uint64_t num_beams, const double* beam_ras,
                                          const double* beam_decs) {
    const uint _num_elements = config.get<int>(unique_name, "num_elements");
    
    const double inst_lat = config.get<double>(unique_name, "inst_lat");
    const double inst_long = config.get<double>(unique_name, "inst_long");    
    const std::vector<float>feed_positions = config.get<std::vector<float>>(unique_name, "feed_positions");
    const double D2R = M_PI / 180.0;
    const double TAU = 2.0 * M_PI;
    const double one_over_c = 3.3356;
    const double phi_0 = 280.46;
    const double lst_rate = 360.0 / 86164.09054;
    const double j2000_unix = 946728000;
    
    double precession_offset = (beamform_time - j2000_unix) * 0.012791 / (365.0 * 24.0 * 3600.0);
    double lst = phi_0 + inst_long + lst_rate * (beamform_time - j2000_unix) - precession_offset;
    lst = fmod(lst, 360.0);
    
    DEBUG("Computing phases for {} beams at time {}", num_beams, beamform_time);

    for (uint64_t beam_idx = 0; beam_idx < num_beams; beam_idx++) {
        // double ra = beam_ras[beam_idx] * D2R;
        double dec = beam_decs[beam_idx] * D2R;
        double hour_angle = (lst - beam_ras[beam_idx]) * D2R;
        
        double alt = sin(dec) * sin(inst_lat)
                   + cos(dec) * cos(inst_lat) * cos(hour_angle);
        alt = asin(alt);
        
        double az = (sin(dec) - sin(alt) * sin(inst_lat))
                  / (cos(alt) * cos(inst_lat));
        az = acos(az);
        if (sin(hour_angle) >= 0) {
            az = TAU - az;
        }
        
        for (size_t elem_idx = 0; elem_idx < _num_elements; elem_idx++) {
            double elem_x = feed_positions[2 * elem_idx];
            double elem_y = feed_positions[2 * elem_idx + 1];
            
            double projection_angle = 90.0 * D2R - atan2(elem_y, elem_x);
            double offset_distance = cos(alt) * sqrt(elem_x * elem_x + elem_y * elem_y);
            double effective_angle = projection_angle - az;
            
            phases[beam_idx * _num_elements + elem_idx] =
                TAU * cos(effective_angle) * offset_distance * one_over_c;
        }
    }
}

basebandDumpData::Status basebandReadout::extract_data(basebandDumpData data) {
    DEBUG("Ready to copy samples into the baseband readout buffer and beamform into baseband_mb readout buffer");
    assert(data.dump_start_frame < data.dump_end_frame);

    const double fpga_period_s = ts_to_double(tel.seq_length());

    const uint64_t event_id = data.event_id;

    int in_buf_frame = data.dump_start_frame % in_buf->num_frames;
    auto first_meta = get_chord_metadata(in_buf, in_buf_frame);

    const uint32_t stream_freq_idx = data.stream_freq_idx;
    const uint32_t freq_id = data.freq_id;
    auto& frame_sent_counter = readout_sent_frame_counter.labels({std::to_string(freq_id)});
    auto& frame_dropped_counter = readout_dropped_frame_counter.labels({std::to_string(freq_id)});

    // Figure out how much data we have.
    int64_t data_start_fpga = std::max(data.trigger_start_fpga, first_meta->get_fpga_seq_num());
    // For now just assume that we have the last sample, because the locking logic
    // currently waits for it. Could be made to be more robust.
    int64_t data_end_fpga = data.trigger_start_fpga + data.trigger_length_fpga;

    timeval tmp, delta;
    delta.tv_sec = 0;
    delta.tv_usec =
        (data.trigger_start_fpga - first_meta->get_fpga_seq_num()) * fpga_period_s * 1e6;
    timeval first_packet_recv_time = first_meta->get_first_packet_recv_time();
    timeradd(&first_packet_recv_time, &delta, &tmp);
    timespec packet_time0 = {tmp.tv_sec, tmp.tv_usec * 1000};

    timespec time0 = tel.to_time(data_start_fpga);
    double ftime0 = ts_to_double(time0);
    double ftime0_offset = (time0.tv_nsec - fmod(ftime0, 1.) * 1e9) / 1e9;

    INFO("Dump data for {:d}/{:d}: frames {:d}-{:d}; samples {}-{}.", event_id, freq_id,
         data.dump_start_frame, data.dump_end_frame, data_start_fpga, data_end_fpga);

    // Number of time samples per output frame
    const int out_frame_samples = out_buf->frame_size / _num_elements;
    
    // Number of time samples per multibeam output frame
    // MB output has shape (samples_per_dataset, num_local_freq, num_beams)
    // So frame_size = samples_per_dataset * num_local_freq * num_beams
    // Thus: frame_samples = frame_size / (num_local_freq * num_beams)
    const int outmb_frame_samples = (_num_beams > 0) ? 
        (outmb_buf->frame_size / (_num_freq_per_stream * _num_beams)) : 0;
    
    // Current frame & metadata in the output buffer
    // Also create pointers to MB readout
    uint8_t* out_frame = nullptr;
    BasebandMetadata* out_metadata = nullptr;
    uint8_t* outmb_frame = nullptr;
    BasebandMetadata* outmb_metadata = nullptr;

    // Available space in the `out_frame` (as interval of length `out_remaining`
    // time_samples, starting at `out_start`)
    // Also keep track of MB readout
    int64_t out_start = 0;
    int64_t out_remaining = 0;
    int64_t outmb_start = 0;
    int64_t outmb_remaining = 0;

    // If the output buffer is full, or we get a shutdown signal (null frame),
    // then instead of continuing to try and get a new output frame for each input frame
    // simple stop extracting data and release all the input frames.
    bool stop_extract = false;

    for (int frame_index = data.dump_start_frame; !stop_thread && frame_index < data.dump_end_frame;
         frame_index++) {

        if (stop_extract) {
            frame_dropped_counter.inc();
            frame_locks[frame_index % _num_frames_buffer].unlock();
            continue;
        }

        in_buf_frame = frame_index % in_buf->num_frames;
        auto metadata = get_chord_metadata(in_buf, in_buf_frame);
        uint8_t* in_buf_data = in_buf->frames[in_buf_frame];
        int64_t frame_fpga_seq = metadata->get_fpga_seq_num();
        int64_t in_start = std::max(data_start_fpga - frame_fpga_seq, (int64_t)0);
        int64_t in_end = std::min(data_end_fpga - frame_fpga_seq, (int64_t)_samples_per_data_set);
        DEBUG("Next input frame: {},  samples {}-{}", frame_index, in_start, in_end);
        while (in_start < in_end) {
	    // Copy frame from input buffer to output buffer.
	    // Track progress by incrementing in_start.
            // Populate metadata for the full-array data
	    if (out_remaining == 0) {
                // Is there an available output frame?
                if (!out_buf->is_frame_empty(out_frame_id)) {
                    // No, skip this frame
                    WARN("Output buffer full ({:d}). Dropping frame {:d}/{:d}", out_frame_id,
                         event_id, frame_index);
                    frame_dropped_counter.inc();
                    stop_extract = true;
                    break;
                }
                // Get a pointer to the new out frame (cannot block because of the check above)
		// CL: ignore checks for outmb buffer...hopefully does not make a difference?
                out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
                outmb_frame = outmb_buf->wait_for_empty_frame(unique_name, outmb_frame_id);
                if (outmb_frame == nullptr) {
                    // Skip this frame
                    WARN("Cannot get an MB output frame ({:d}). Dropping frame {:d}/{:d}",
                         outmb_frame_id, event_id, frame_index);
                    frame_dropped_counter.inc();
                    stop_extract = true;
                    break;
                }
                if (out_frame == nullptr) {
                    // Skip this frame
                    WARN("Cannot get an output frame ({:d}). Dropping frame {:d}/{:d}",
                         out_frame_id, event_id, frame_index);
                    frame_dropped_counter.inc();
                    stop_extract = true;
                    break;
                }
		// Metadata for this frame, standard path
                out_start = 0;
                out_remaining = out_frame_samples;
                
                // Initialize multibeam output frame tracking
                outmb_start = 0;
                outmb_remaining = outmb_frame_samples;

                out_buf->allocate_new_metadata_object(out_frame_id);
                out_metadata = (BasebandMetadata*)(out_buf->get_metadata(out_frame_id).get());

                out_metadata->event_id = event_id;
                out_metadata->freq_id = freq_id;
                out_metadata->event_start_fpga = data.trigger_start_fpga;
                out_metadata->event_end_fpga = data.trigger_start_fpga + data.trigger_length_fpga;

                out_metadata->time0_fpga = data_start_fpga;
                out_metadata->time0_ctime = ftime0;
                out_metadata->time0_ctime_offset = ftime0_offset;

                out_metadata->first_packet_recv_time = ts_to_double(packet_time0);
                out_metadata->fpga0_ns = fpga0_ns;
                out_metadata->frame_fpga_seq = frame_fpga_seq + in_start;
                out_metadata->valid_to = 0; // gets adjusted as we copy the data
                out_metadata->num_elements = _num_elements;
                out_metadata->reserved = -1;
                
		// Metadata for the beamformed frame, nonstandard path
		outmb_buf->allocate_new_metadata_object(outmb_frame_id);
                outmb_metadata = (BasebandMetadata*)(outmb_buf->get_metadata(outmb_frame_id).get());

                outmb_metadata->event_id = event_id;
                outmb_metadata->freq_id = freq_id;
                outmb_metadata->event_start_fpga = data.trigger_start_fpga;
                outmb_metadata->event_end_fpga = data.trigger_start_fpga + data.trigger_length_fpga;

                outmb_metadata->time0_fpga = data_start_fpga;
                outmb_metadata->time0_ctime = ftime0;
                outmb_metadata->time0_ctime_offset = ftime0_offset;

                outmb_metadata->first_packet_recv_time = ts_to_double(packet_time0);
                outmb_metadata->fpga0_ns = fpga0_ns;
                outmb_metadata->frame_fpga_seq = frame_fpga_seq + in_start;
                outmb_metadata->valid_to = 0; // gets adjusted as we copy the data
                outmb_metadata->num_elements = _num_elements;
                outmb_metadata->reserved = -1;
            }

            // copy the data
            int64_t copy_len;
            if (_num_freq_per_stream == 1) {
                copy_len = std::min(in_end - in_start, out_remaining);
                DEBUG("Copy samples {}/{}-{} to {}/{} ({} bytes)", frame_index, in_start,
                      in_start + copy_len * _num_elements, out_frame_id, out_start,
                      copy_len * _num_elements);
                memcpy(out_frame + (out_start * _num_elements),
                       in_buf_data + (in_start * _num_elements), copy_len * _num_elements);
                
                // TODO: Implement actual multibeam beamforming here
                // For now, zero-fill the multibeam output buffer to maintain correct geometry
                if (_num_beams > 0) {
                    const int64_t outmb_copy_len = copy_len * _num_beams;
                    memset(outmb_frame + (outmb_start * _num_beams), 0, outmb_copy_len);
                    DEBUG("Multibeam placeholder fill (single-freq): {} bytes at offset {}", 
                          outmb_copy_len, outmb_start * _num_beams);
                }
            } else {
                copy_len = std::min((int64_t)1, out_remaining);
                DEBUG("Copy samples {}/{}-{} for in-frame frequency {} to {}/{} ({} bytes, "
                      "starting at {})",
                      frame_index, in_start, in_start + copy_len * _num_elements, stream_freq_idx,
                      out_frame_id, out_start, copy_len * _num_elements,
                      (in_start * _num_freq_per_stream + stream_freq_idx) * _num_elements);
                memcpy(out_frame + (out_start * _num_elements),
                       in_buf_data
                           + (in_start * _num_freq_per_stream + stream_freq_idx) * _num_elements,
                       copy_len * _num_elements);
                
                // TODO: Implement actual multibeam beamforming here
                // For now, zero-fill the multibeam output buffer to maintain correct geometry
                // Actual beamforming will: unpack 4-bit data, apply phases & gains, sum over elements, quantize
                if (_num_beams > 0) {
                    // Output geometry: (time_samples, num_local_freq, num_beams)
                    // Each entry is 1 byte (int4x2_t packing real+imag)
                    const int64_t outmb_copy_len = copy_len * _num_freq_per_stream * _num_beams;
                    memset(outmb_frame + (outmb_start * _num_freq_per_stream * _num_beams), 0, outmb_copy_len);
                    DEBUG("Multibeam placeholder fill: {} bytes at offset {}", 
                          outmb_copy_len, outmb_start * _num_freq_per_stream * _num_beams);
                }
            }
            in_start += copy_len;
            out_start += copy_len;
            outmb_start += copy_len;
            out_metadata->valid_to = out_start;
            outmb_metadata->valid_to = out_start;
            out_remaining -= copy_len;
            outmb_remaining -= copy_len;
            if (out_remaining == 0) {
                out_buf->mark_frame_full(unique_name, out_frame_id++);
                frame_sent_counter.inc();
            }
            if (outmb_remaining == 0) {
                outmb_buf->mark_frame_full(unique_name, outmb_frame_id++);
                frame_sent_counter.inc();
            }
        }

        // Done with this frame. Allow it to participate in the ring buffer.
        frame_locks[frame_index % _num_frames_buffer].unlock();
    }

    // after all input frames are done, flush the out frame if it's incomplete:
    if (out_remaining > 0) {
        DEBUG("Clearing out the remaining {} samples of the frame: {}/{} ({} bytes)", out_remaining,
              out_frame_id, out_start, (out_remaining * _num_elements));
        memset(out_frame + (out_start * _num_elements), 0, out_remaining * _num_elements);
        memset(outmb_frame + (outmb_start * NUM_BEAMS), 0, out_remaining * NUM_BEAMS);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
	outmb_frame_id++;
        frame_sent_counter.inc();
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
