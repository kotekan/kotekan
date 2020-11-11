#include "pulsarPostProcess.hpp"

#include "BranchPrediction.hpp" // for likely, unlikely
#include "Config.hpp"           // for Config
#include "ICETelescope.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.h"             // for Buffer, mark_frame_empty, wait_for_empty_frame, wait_...
#include "bufferContainer.hpp"  // for bufferContainer
#include "chimeMetadata.h"      // for get_fpga_seq_num, beamCoord, get_beam_coord, get_stream...
#include "kotekanLogging.hpp"   // for DEBUG, ERROR
#include "pulsar_functions.hpp" // for PSRHeader

#include <algorithm>  // for max
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cmath>      // for round
#include <cstdint>    // for int64_t, uint64_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <string.h>   // for memcpy
#include <string>     // for allocator, string, operator+, to_string
#include <vector>     // for vector

using std::string;

#define udp_pulsar_header_size 32


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(pulsarPostProcess);

pulsarPostProcess::pulsarPostProcess(Config& config_, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&pulsarPostProcess::main_thread, this)) {

    // Apply config.
    _num_gpus = config.get<uint32_t>(unique_name, "num_gpus");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _num_pulsar = config.get<uint32_t>(unique_name, "num_beams");
    _num_pol = config.get<uint32_t>(unique_name, "num_pol");
    _timesamples_per_pulsar_packet =
        config.get<uint32_t>(unique_name, "timesamples_per_pulsar_packet");
    _udp_pulsar_packet_size = config.get<uint32_t>(unique_name, "udp_pulsar_packet_size");
    _num_packet_per_stream = config.get<uint32_t>(unique_name, "num_packet_per_stream");
    _num_stream = config.get<uint32_t>(unique_name, "num_stream");

    assert(_timesamples_per_pulsar_packet == 625 || _timesamples_per_pulsar_packet == 3125);

    in_buf = new Buffer*[_num_gpus];
    in_buffer_ID = new uint[_num_gpus]; // 4 of these , cycle through buffer depth
    in_frame = new uint8_t*[_num_gpus];

    for (uint32_t i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer("network_input_buffer_" + std::to_string(i));
        register_consumer(in_buf[i], unique_name.c_str());
    }
    pulsar_buf = get_buffer("pulsar_out_buf");
    register_producer(pulsar_buf, unique_name.c_str());
}

pulsarPostProcess::~pulsarPostProcess() {
    delete[] in_buf;
    delete[] in_buffer_ID;
    delete[] in_frame;
}

void pulsarPostProcess::fill_headers(unsigned char* out_buf, PSRHeader* psr_header,
                                     const uint64_t fpga_seq_num, timespec* time_now,
                                     beamCoord* beam_coord, uint16_t* thread_ids) {

    // Get the Telescope instance and pre-calc the length of an FPGA frame
    auto& tel = Telescope::instance();
    const uint64_t fpga_ns = tel.seq_length_nsec();
    const float fpga_s = 1e-9 * fpga_ns;

    uint freqloop = _num_stream / _num_pulsar;
    DEBUG("Filling headers starting at {} ({}.{:09d})", fpga_seq_num, time_now->tv_sec,
          time_now->tv_nsec);
    for (uint i = 0; i < _num_packet_per_stream; ++i) { // 16 or 80 frames in a stream
        uint64_t fpga_now = (fpga_seq_num + _timesamples_per_pulsar_packet * i);
        psr_header->seconds = time_now->tv_sec - unix_offset;
        psr_header->data_frame =
            (time_now->tv_nsec / 1.e9) / (_timesamples_per_pulsar_packet * fpga_s);

        for (uint f = 0; f < freqloop; ++f) {
            // Load frequency indices into thread_id and EUD3 if frequency packing.
            psr_header->thread_id = thread_ids[f];
            if (_timesamples_per_pulsar_packet == 625) {
                psr_header->eud3 = (uint32_t)thread_ids[1] + ((uint32_t)thread_ids[2] << 10)
                                   + ((uint32_t)thread_ids[3] << 20);
            }

            for (uint32_t beam_id = 0; beam_id < _num_pulsar; ++beam_id) {
                psr_header->eud1 = beam_id; // beam id
                psr_header->eud2 = beam_coord[f].scaling[beam_id];
                uint16_t ra_part = (uint16_t)(beam_coord[f].ra[beam_id] * 100);
                uint16_t dec_part = (uint16_t)((beam_coord[f].dec[beam_id] + 90) * 100);
                psr_header->eud4 = (ra_part << 16) + (dec_part & 0xFFFF);
                timespec time_now_from_compute = tel.to_time(fpga_now);
                if (time_now->tv_sec != time_now_from_compute.tv_sec) {
                    ERROR("[Time Check] mismatch in fill header packet={:d} beam={:d} "
                          "time_now->tv_sec={:d} time_now_from_compute.tv_sec={:d}",
                          i, beam_id, time_now->tv_sec, time_now_from_compute.tv_sec);
                }
                if (time_now->tv_nsec != time_now_from_compute.tv_nsec) {
                    ERROR("[Time Check] mismatch in fill header packet={:d} beam={:d} "
                          "time_now->tv_nsec={:d} time_now_from_compute.tv_nsec={:d}",
                          i, beam_id, time_now->tv_nsec, time_now_from_compute.tv_nsec);
                }
                if (_timesamples_per_pulsar_packet == 3125) {
                    memcpy(&out_buf[(f * _num_pulsar + beam_id) * _num_packet_per_stream
                                        * _udp_pulsar_packet_size
                                    + i * _udp_pulsar_packet_size],
                           psr_header, sizeof(PSRHeader));
                } else if (_timesamples_per_pulsar_packet == 625) {
                    memcpy(&out_buf[beam_id * _num_packet_per_stream * _udp_pulsar_packet_size
                                    + i * _udp_pulsar_packet_size],
                           psr_header, sizeof(PSRHeader));
                }
            }
        } // end freq
        // Increment time for the next frame
        time_now->tv_nsec += _timesamples_per_pulsar_packet * fpga_ns;
        if (time_now->tv_nsec > 999999999) {
            time_now->tv_sec += (uint)(time_now->tv_nsec / 1000000000.);
            time_now->tv_nsec = time_now->tv_nsec % 1000000000;
        }
    } // end packet
}

void pulsarPostProcess::main_thread() {

    auto& tel = Telescope::instance();
    uint64_t fpga_ns = tel.seq_length_nsec();

    int out_buffer_ID = 0;
    int startup = 1; // related to the likely & unlikely
    uint16_t thread_ids[_num_gpus];

    for (uint32_t i = 0; i < _num_gpus; ++i) {
        in_buffer_ID[i] = 0;
    }
    uint64_t frame_fpga_seq_num = 0;     // sample starting the current input frame
    uint32_t current_input_location = 0; // goes from 0 to _samples_per_data_set

    PSRHeader psr_header;
    psr_header.seconds = 0; // UD
    psr_header.legacy = 0;
    psr_header.invalid = 0;
    psr_header.data_frame = 0; // UD
    psr_header.ref_epoch = 36; // First half of 2018.
    unix_offset = 1514764800;  // corresponds to 2018.01.01.0:0:0 in UTC
    psr_header.unused = 0;
    if (_timesamples_per_pulsar_packet == 3125) {
        psr_header.frame_len = 768;  //(6250-B data + 6-B pad + 32-B header)
        psr_header.log_num_chan = 1; // 2pol so ln2=1
    } else if (_timesamples_per_pulsar_packet == 625) {
        psr_header.frame_len = 629;  // 5032-B
        psr_header.log_num_chan = 3; // ln8
    }
    psr_header.vdif_version = 1;
    psr_header.station_id = 0; // to be set as a node ID after buffer sync, see below.
    psr_header.thread_id = 0;  // index of first packed frequency.
    psr_header.bits_depth = 3; // 4+4 bit so 4-1=3
    psr_header.data_type = 1;  // Complex
    psr_header.edv = 0;
    psr_header.eud1 = 0; // UD: beam number [0 to 9]
    psr_header.eud2 = 0; //_psr_scaling from metadata
    psr_header.eud3 = 0; // number computed using bitwise-shifted frequency indeces.
    psr_header.eud4 = 0; // 16-b RA + 16-b Dec

    uint frame = 0;
    uint in_frame_location = 0; // goes from 0 to 3125 or 625
    uint64_t fpga_seq_num = 0;

    beamCoord beam_coord[_num_gpus];
    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = wait_for_empty_frame(pulsar_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == nullptr)
        goto end_loop;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        auto new_frame_fpga_seq_num = sync_input_buffers();
        if (!new_frame_fpga_seq_num.has_value())
            return;

        // Initialize data for header info, namely position and frequency labels.
        for (uint32_t i = 0; i < _num_gpus; ++i) {
            beam_coord[i] = get_beam_coord(in_buf[i], in_buffer_ID[i]);
            thread_ids[i] = tel.to_freq_id(in_buf[i], in_buffer_ID[i]);
        }

        // Define station_id as a node identifer in terms of F-engine slot/crate/link data.
        ice_stream_id_t stream_id = ice_get_stream_id_t(in_buf[0], in_buffer_ID[0]);
        psr_header.station_id =
            (uint16_t)(stream_id.crate_id * 16 + stream_id.slot_id + stream_id.link_id * 32);

        bool skipped_frames =
            (new_frame_fpga_seq_num.value() - frame_fpga_seq_num) > _samples_per_data_set;
        frame_fpga_seq_num = new_frame_fpga_seq_num.value();

        // If this is the first time wait until we get the start of an interger second period.
        if (unlikely(startup == 1)) {
            startup = 0;

            if (!tel.gps_time_enabled()) {
                ERROR("[Time Check] gps global time is not set.");
            }
            time_now = tel.to_time(frame_fpga_seq_num);
            uint32_t pkt_length_in_ns = _timesamples_per_pulsar_packet * fpga_ns;
            uint32_t ns_offset = pkt_length_in_ns - (time_now.tv_nsec % pkt_length_in_ns);
            float seq_number_offset_float = ns_offset / (float)fpga_ns;
            uint seq_number_offset = round(seq_number_offset_float);

            current_input_location = seq_number_offset;
            fpga_seq_num = frame_fpga_seq_num + seq_number_offset;
            time_now.tv_nsec += seq_number_offset * fpga_ns;
            while (time_now.tv_nsec >= 1e9) {
                time_now.tv_sec += 1;
                time_now.tv_nsec -= 1e9;
            }
            skipped_frames = false;

            // Fill the first output buffer headers
            fill_headers((unsigned char*)out_frame, &psr_header, fpga_seq_num, &time_now,
                         beam_coord, thread_ids);
        }

        // Take data from the input buffer and format the output
        if (likely(startup == 0)) {
            // Adjust the output position and in-frame time when we skip any input frames
            if (skipped_frames) {
                // TODO this section duplicates the code in startup, look into combining them
                fpga_seq_num = frame_fpga_seq_num;
                DEBUG("Skipped frames to {}; current output packet {} / {}", fpga_seq_num, frame,
                      in_frame_location);
                time_now = tel.to_time(fpga_seq_num);
                DEBUG("GPS clock {}.{:09d}, fpga_seq_num: {}", time_now.tv_sec, time_now.tv_nsec,
                      fpga_seq_num);

                uint32_t pkt_length_in_ns = _timesamples_per_pulsar_packet * fpga_ns;
                uint32_t ns_offset = pkt_length_in_ns - (time_now.tv_nsec % pkt_length_in_ns);
                float seq_number_offset_float = ns_offset / (float)fpga_ns;
                uint seq_number_offset = round(seq_number_offset_float);

                current_input_location = seq_number_offset;
                fpga_seq_num += seq_number_offset;
                time_now.tv_nsec += seq_number_offset * fpga_ns;
                while (time_now.tv_nsec >= 1e9) {
                    time_now.tv_sec += 1;
                    time_now.tv_nsec -= 1e9;
                }
                DEBUG("Advanced the clock to {}.{:09d}", time_now.tv_sec, time_now.tv_nsec);

                frame = 0;
                in_frame_location = 0;

                // Fill the headers of the new buffer
                fill_headers((unsigned char*)out_frame, &psr_header, fpga_seq_num, &time_now,
                             beam_coord, thread_ids);
            }

            // now store samples into output buffer.
            for (uint32_t thread_id = 0; thread_id < _num_gpus; ++thread_id) {
                float* in_buf_data = (float*)in_frame[thread_id];

                for (uint i = current_input_location; i < _samples_per_data_set; ++i) {
                    if (in_frame_location == _timesamples_per_pulsar_packet) { // last sample
                        in_frame_location = 0;
                        frame++;
                        if (frame == _num_packet_per_stream) { // last frame
                            frame = 0;
                            mark_frame_full(pulsar_buf, unique_name.c_str(), out_buffer_ID);
                            // Get a new output buffer
                            out_buffer_ID = (out_buffer_ID + 1) % pulsar_buf->num_frames;
                            out_frame =
                                wait_for_empty_frame(pulsar_buf, unique_name.c_str(),
                                                     out_buffer_ID);
                            if (out_frame == nullptr)
                                goto end_loop;
                            // Fill the headers of the new buffer
                            fpga_seq_num += _timesamples_per_pulsar_packet * _num_packet_per_stream;
                            fill_headers((unsigned char*)out_frame, &psr_header, fpga_seq_num,
                                         &time_now, beam_coord, thread_ids);
                        } // end if last frame
                    }     // end if last sample

                    unsigned char* out_buf = (unsigned char*)out_frame;

                    for (uint32_t psr = 0; psr < _num_pulsar; ++psr) { // loop psr
                        for (uint32_t p = 0; p < _num_pol; ++p) {
                            uint32_t out_index = 0;
                            if (_timesamples_per_pulsar_packet == 3125) {
                                // freq->beam->packets->[time-pol]
                                out_index = (thread_id * _num_pulsar + psr)
                                                * _udp_pulsar_packet_size * _num_packet_per_stream
                                            + frame * _udp_pulsar_packet_size
                                            + (in_frame_location * _num_pol + p)
                                            + udp_pulsar_header_size;
                            } else if (_timesamples_per_pulsar_packet == 625) {
                                // beam->packets->[freq-time-pol]
                                out_index = psr * _udp_pulsar_packet_size * _num_packet_per_stream
                                            + frame * _udp_pulsar_packet_size
                                            + (thread_id * _samples_per_data_set * _num_pol
                                               + in_frame_location * _num_pol + p)
                                            + udp_pulsar_header_size;
                            } else
                                throw std::runtime_error("Unknown timesamples per VDIF packet.");

                            // clang-format off
                            float real_float =
                                ((in_buf_data[(i * _num_pulsar * _num_pol + psr * _num_pol + p) * 2])
                                / float(beam_coord[thread_id].scaling[psr]) + 0.5) + 8;
                            float imag_float =
                                ((in_buf_data[(i * _num_pulsar * _num_pol + psr * _num_pol + p) * 2 + 1])
                                / float(beam_coord[thread_id].scaling[psr]) + 0.5) + 8;
                            // clang-format on
                            if (real_float > 15)
                                real_float = 15.;
                            if (imag_float > 15)
                                imag_float = 15.;
                            if (real_float < 0)
                                real_float = 0.;
                            if (imag_float < 0)
                                imag_float = 0.;
                            uint8_t real_part = int(real_float);
                            uint8_t imag_part = int(imag_float);

                            out_buf[out_index] = ((real_part << 4) & 0xF0) + (imag_part & 0x0F);
                        } // end loop pol
                    }     // end loop psr
                    in_frame_location++;
                } // end loop time
            }     // end loop freq
            current_input_location = 0;
        } // end if not start up

        // Release the input buffers
        for (uint32_t i = 0; i < _num_gpus; ++i) {
            // release_info_object(in_buf[gpu_id], in_buffer_ID[i]);
            mark_frame_empty(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
            in_buffer_ID[i] = (in_buffer_ID[i] + 1) % in_buf[i]->num_frames;
        }
    } // end stop thread
end_loop:;
}


std::optional<uint64_t> pulsarPostProcess::sync_input_buffers() {
    for (unsigned i = 0; i < _num_gpus; i++) {
        in_frame[i] = wait_for_full_frame(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
        if (in_frame[i] == nullptr)
            return std::nullopt;
    }
    while (!stop_thread) {
        // Get all input buffers in sync by fpga_seq_no: find the one that's the
        // furthest along, and keep advancing others until they all match. (Keep in
        // mind that advancing one of the others may put it ahead of the current
        // largest fpga_seq_no, in which case we have to repeat the process.)
        auto max_fpga_count = get_fpga_seq_num(in_buf[0], in_buffer_ID[0]);
        for (unsigned i = 1; i < _num_gpus; i++) {
            max_fpga_count = std::max(max_fpga_count, get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));
        }
        bool fpga_seq_in_sync = true;
        for (unsigned i = 0; i < _num_gpus; ++i) {
            while (max_fpga_count > get_fpga_seq_num(in_buf[i], in_buffer_ID[i])) {
                mark_frame_empty(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
                in_buffer_ID[i] = (in_buffer_ID[i] + 1) % in_buf[i]->num_frames;
                in_frame[i] = wait_for_full_frame(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
                if (in_frame[i] == nullptr)
                    return std::nullopt;
            }
            if (max_fpga_count != get_fpga_seq_num(in_buf[i], in_buffer_ID[i])) {
                fpga_seq_in_sync = false;
            }
        }
        if (fpga_seq_in_sync) {
            break;
        }
    }
    if (stop_thread)
        return std::nullopt;

    return get_fpga_seq_num(in_buf[0], in_buffer_ID[0]);
}
