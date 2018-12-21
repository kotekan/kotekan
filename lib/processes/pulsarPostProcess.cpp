#include "gpsTime.h"

#include <assert.h>
#include <functional>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <string>

using std::string;

// TODO Where do these live?
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define udp_pulsar_header_size 32

#include "Config.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "pulsarPostProcess.hpp"
#include "util.h"
#include "vdif_functions.h"

REGISTER_KOTEKAN_PROCESS(pulsarPostProcess);

pulsarPostProcess::pulsarPostProcess(Config& config_, const string& unique_name,
                                     bufferContainer& buffer_container) :
    KotekanProcess(config_, unique_name, buffer_container,
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

    in_buf = (struct Buffer**)malloc(_num_gpus * sizeof(struct Buffer*));
    for (uint32_t i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer("network_input_buffer_" + std::to_string(i));
        register_consumer(in_buf[i], unique_name.c_str());
    }
    pulsar_buf = get_buffer("pulsar_out_buf");
    register_producer(pulsar_buf, unique_name.c_str());
}

pulsarPostProcess::~pulsarPostProcess() {
    free(in_buf);
}

void pulsarPostProcess::fill_headers(unsigned char* out_buf, struct VDIFHeader* vdif_header,
                                     const uint64_t fpga_seq_num, struct timespec* time_now,
                                     struct psrCoord* psr_coord, uint16_t* freq_ids) {
    uint freqloop = _num_stream / _num_pulsar;
    for (uint i = 0; i < _num_packet_per_stream; ++i) { // 16 or 80 frames in a stream
        uint64_t fpga_now = (fpga_seq_num + _timesamples_per_pulsar_packet * i);
        vdif_header->eud3 = (fpga_now & (0xFFFFFFFFl << 0)) >> 0;
        vdif_header->seconds = time_now->tv_sec - unix_offset;
        vdif_header->data_frame =
            (time_now->tv_nsec / 1.e9) / (_timesamples_per_pulsar_packet * 2.56e-6);

        for (uint f = 0; f < freqloop; ++f) {
            vdif_header->thread_id = freq_ids[f];

            for (uint32_t psr = 0; psr < _num_pulsar; ++psr) {
                vdif_header->eud1 = psr; // beam id
                vdif_header->eud2 = psr_coord[f].scaling[psr];
                uint16_t ra_part = (uint16_t)(psr_coord[f].ra[psr] * 100);
                uint16_t dec_part = (uint16_t)((psr_coord[f].dec[psr] + 90) * 100);
                vdif_header->eud4 = ((ra_part << 16) & 0xFFFF0000) + (dec_part & 0xFFFF);
                struct timespec time_now_from_compute;
                time_now_from_compute = compute_gps_time(fpga_now);
                if (time_now->tv_sec != time_now_from_compute.tv_sec) {
                    ERROR("[Time Check] mismatch in fill header packet=%d beam=%d "
                          "time_now->tv_sec=%ld time_now_from_compute.tv_sec=%ld",
                          i, psr, time_now->tv_sec, time_now_from_compute.tv_sec);
                }
                if (time_now->tv_nsec != time_now_from_compute.tv_nsec) {
                    ERROR("[Time Check] mismatch in fill header packet=%d beam=%d "
                          "time_now->tv_nsec=%ld time_now_from_compute.tv_nsec=%ld",
                          i, psr, time_now->tv_nsec, time_now_from_compute.tv_nsec);
                }
                if (_timesamples_per_pulsar_packet == 3125) {
                    memcpy(&out_buf[(f * _num_pulsar + psr) * _num_packet_per_stream
                                        * _udp_pulsar_packet_size
                                    + i * _udp_pulsar_packet_size],
                           vdif_header, sizeof(struct VDIFHeader));
                } else if (_timesamples_per_pulsar_packet == 625) {
                    memcpy(&out_buf[psr * _num_packet_per_stream * _udp_pulsar_packet_size
                                    + i * _udp_pulsar_packet_size],
                           vdif_header, sizeof(struct VDIFHeader));
                }
            }
        } // end freq
        // Increment time for the next frame
        time_now->tv_nsec += _timesamples_per_pulsar_packet * 2560;
        if (time_now->tv_nsec > 999999999) {
            time_now->tv_sec += (uint)(time_now->tv_nsec / 1000000000.);
            time_now->tv_nsec = time_now->tv_nsec % 1000000000;
        }
    } // end packet
}

void pulsarPostProcess::main_thread() {

    uint in_buffer_ID[_num_gpus]; // 4 of these , cycle through buffer depth
    uint8_t* in_frame[_num_gpus];
    int out_buffer_ID = 0;
    int startup = 1; // related to the likely & unlikely
    uint16_t freq_ids[_num_gpus];

    for (uint32_t i = 0; i < _num_gpus; ++i) {
        in_buffer_ID[i] = 0;
    }
    uint32_t current_input_location = 0; // goes from 0 to _samples_per_data_set

    struct VDIFHeader vdif_header;
    vdif_header.seconds = 0; // UD
    vdif_header.legacy = 0;
    vdif_header.invalid = 0;
    vdif_header.data_frame = 0; // UD
    vdif_header.ref_epoch = 36; // First half of 2018.
    unix_offset = 1514764800;   // corresponds to 2018.01.01.0:0:0 in UTC
    vdif_header.unused = 0;
    if (_timesamples_per_pulsar_packet == 3125) {
        vdif_header.frame_len = 768;  //(6250-B data + 6-B pad + 32-B header)
        vdif_header.log_num_chan = 1; // 2pol so ln2=1
    } else if (_timesamples_per_pulsar_packet == 625) {
        vdif_header.frame_len = 629;  // 5032-B
        vdif_header.log_num_chan = 3; // ln8
    }
    vdif_header.vdif_version = 1;
    char si[2] = {'C', 'X'};
    vdif_header.station_id = (si[0] << 8) + si[1]; // Need to fomrally ask the Vdif community
    vdif_header.thread_id = 0;                     // UD   freq
    vdif_header.bits_depth = 3;                    // 4+4 bit so 4-1=3
    vdif_header.data_type = 1;                     // Complex
    vdif_header.edv = 0;
    vdif_header.eud1 = 0; // UD: beam number [0 to 9]
    vdif_header.eud2 = 0; //_psr_scaling from metadata
    vdif_header.eud3 = 0; // UD: fpga count low bit
    vdif_header.eud4 = 0; // 16-b RA + 16-b Dec

    uint frame = 0;
    uint in_frame_location = 0; // goes from 0 to 3125 or 625
    uint64_t fpga_seq_num = 0;

    struct psrCoord psr_coord[_num_gpus];
    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = wait_for_empty_frame(pulsar_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == NULL)
        goto end_loop;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        for (uint32_t i = 0; i < _num_gpus; ++i) {
            in_frame[i] = wait_for_full_frame(in_buf[i], unique_name.c_str(), in_buffer_ID[i]);
            if (in_frame[i] == NULL)
                goto end_loop;

            psr_coord[i] = get_psr_coord(in_buf[i], in_buffer_ID[i]);
        }
        uint64_t first_seq_number = get_fpga_seq_num(in_buf[0], in_buffer_ID[0]);

        for (uint32_t i = 0; i < _num_gpus; ++i) {
            assert(first_seq_number == (uint64_t)get_fpga_seq_num(in_buf[i], in_buffer_ID[i]));

            stream_id_t stream_id = get_stream_id_t(in_buf[i], in_buffer_ID[i]);
            freq_ids[i] = bin_number_chime(&stream_id);
        }


        // If this is the first time wait until we get the start of an interger second period.
        if (unlikely(startup == 1)) {
            // testing sync code
            startup = 0;

            // GPS time, need ch_master
            time_now = get_gps_time(in_buf[0], in_buffer_ID[0]);

            struct timespec time_now_from_compute2 = compute_gps_time(first_seq_number);
            if (time_now.tv_sec != time_now_from_compute2.tv_sec) {
                ERROR("[Time Check] mismatch in execute time_now.tv_sec=%ld "
                      "time_now_from_compute2.tv_sec=%ld",
                      time_now.tv_sec, time_now_from_compute2.tv_sec);
            }
            if (time_now.tv_nsec != time_now_from_compute2.tv_nsec) {
                ERROR("[Time Check] mismatch in execute time_now.tv_nsec=%ld "
                      "time_now_from_compute2.tv_nsec=%ld",
                      time_now.tv_nsec, time_now_from_compute2.tv_nsec);
            }
            if (is_gps_global_time_set() != 1) {
                ERROR("[Time Check] gps global time not set (%d)", is_gps_global_time_set());
            }
            uint32_t seq_number_offset = _timesamples_per_pulsar_packet
                                         - (first_seq_number % _timesamples_per_pulsar_packet);
            current_input_location = seq_number_offset;
            first_seq_number =
                first_seq_number + seq_number_offset; // so that we start at an fpga_seq_no that is
                                                      // divisible by the packet nsamp
            time_now.tv_nsec += seq_number_offset * 2560;
            if (time_now.tv_nsec > 999999999) {
                time_now.tv_sec += (uint)(time_now.tv_nsec / 1000000000.);
                time_now.tv_nsec = time_now.tv_nsec % 1000000000;
            }

            // Fill the first output buffer headers
            fpga_seq_num = first_seq_number;
            fill_headers((unsigned char*)out_frame, &vdif_header, first_seq_number, &time_now,
                         psr_coord, (uint16_t*)freq_ids);
        }
        // This loop which takes data from the input buffer and formats the output.
        if (likely(startup == 0)) {
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
                            wait_for_empty_frame(pulsar_buf, unique_name.c_str(), out_buffer_ID);
                        if (out_frame == NULL)
                            goto end_loop;
                        // Fill the headers of the new buffer
                        fpga_seq_num += _timesamples_per_pulsar_packet * _num_packet_per_stream;
                        fill_headers((unsigned char*)out_frame, &vdif_header, fpga_seq_num,
                                     &time_now, psr_coord, (uint16_t*)freq_ids);
                    } // end if last frame
                }     // end if last sample

                unsigned char* out_buf = (unsigned char*)out_frame;
                for (uint32_t thread_id = 0; thread_id < _num_gpus;
                     ++thread_id) { // loop the 4 GPUs (input)
                    float* in_buf_data = (float*)in_frame[thread_id];
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
                                // beam->packets->[time-freq-pol]
                                out_index = psr * _udp_pulsar_packet_size * _num_packet_per_stream
                                            + frame * _udp_pulsar_packet_size
                                            + (in_frame_location * _num_gpus * _num_pol
                                               + thread_id * _num_pol + p)
                                            + udp_pulsar_header_size;
                            } else
                                throw std::runtime_error("Unknown timesamples per VDIF packet.");

                            uint8_t real_part =
                                int((in_buf_data[(i * _num_pulsar * _num_pol + psr * _num_pol + p)
                                                 * 2])
                                        / float(psr_coord[thread_id].scaling[psr])
                                    + 0.5)
                                + 8;
                            uint8_t imag_part =
                                int((in_buf_data[(i * _num_pulsar * _num_pol + psr * _num_pol + p)
                                                     * 2
                                                 + 1])
                                        / float(psr_coord[thread_id].scaling[psr])
                                    + 0.5)
                                + 8;
                            if (real_part > 15)
                                real_part = 15;
                            if (imag_part > 15)
                                imag_part = 15;
                            out_buf[out_index] = ((real_part << 4) & 0xF0) + (imag_part & 0x0F);
                        } // end loop pol
                    }     // end loop psr
                }         // end loop 4 GPUs
                in_frame_location++;
            } // end looping i
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
