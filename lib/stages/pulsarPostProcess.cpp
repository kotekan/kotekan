#include "pulsarPostProcess.hpp"

#include "BranchPrediction.hpp" // for likely, unlikely
#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"        // for Telescope
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "chordMetadata.hpp"    // for get_chord_metadata, chordMetadata
#include "kotekanLogging.hpp"   // for DEBUG, ERROR
#include "pulsar_functions.hpp" // for PSRHeader

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>  // for max
#include <assert.h>   // for assert
#include <cmath>      // for round
#include <functional> // for bind, function
#include <memory>     // for __shared_ptr_access, shared_ptr
#include <stdexcept>  // for runtime_error
#include <string.h>   // for memcpy
#include <string>     // for allocator, basic_string, operator+, string, to_string
#include <vector>     // for vector

using std::string;

#define udp_pulsar_header_size 32

// this cannot be made into a runtime parameter very easily since it is
// limited by how many frequencies we can stuff into a single 32bit field of
// the VDIF header
#define num_freqs_per_frame 4 // at least for the 625 times version we use


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(pulsarPostProcess);

pulsarPostProcess::pulsarPostProcess(Config& config_, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&pulsarPostProcess::main_thread, this)) {

    // Apply config.
    _num_freqs_per_gpu = config.get_default<uint32_t>(unique_name, "num_freqs_per_gpu", 16);
    assert(_num_freqs_per_gpu % num_freqs_per_frame == 0);
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _num_pulsar_beams = config.get<uint32_t>(unique_name, "num_pulsar_beams");
    _num_beams = config.get<uint32_t>(unique_name, "num_beams");
    _num_pol = config.get<uint32_t>(unique_name, "num_pol");
    _timesamples_per_pulsar_packet =
        config.get<uint32_t>(unique_name, "timesamples_per_pulsar_packet");
    _udp_pulsar_packet_size = config.get<uint32_t>(unique_name, "udp_pulsar_packet_size");
    _num_packet_per_stream = config.get<uint32_t>(unique_name, "num_packet_per_stream");
    _num_stream = config.get<uint32_t>(unique_name, "num_stream");

    // RH: only 625 used in production CHIME
    // AR says to remove the option altogether
    assert(_timesamples_per_pulsar_packet == 625);
    assert(_num_stream == 10);
    assert(num_freqs_per_frame == 4);
    assert(_num_pol == 2);

    // TODO: check chord metadata object to ensure we receive what we expect

    in_buffer = get_buffer("network_input_buffer");
    in_buffer->register_consumer(unique_name);
    pulsar_buf = get_buffer("pulsar_out_buf");
    pulsar_buf->register_producer(unique_name);
}

pulsarPostProcess::~pulsarPostProcess() {}

void pulsarPostProcess::fill_headers(unsigned char* out_buf, PSRHeader* psr_header,
                                     const uint64_t fpga_seq_num, timespec* const time_now,
                                     const chordMetadata::beamCoord& beam_coord,
                                     uint16_t const* const thread_ids) {

    // Get the Telescope instance and pre-calc the length of an FPGA frame
    auto& tel = Telescope::instance();
    const uint64_t fpga_ns = tel.seq_length_nsec();
    const float fpga_s = 1e-9 * fpga_ns;
    const uint32_t unix_offset = 1514764800; // corresponds to 2018.01.01.0:0:0 in UTC

    DEBUG("Filling headers starting at {} ({}.{:09d})", fpga_seq_num, time_now->tv_sec,
          time_now->tv_nsec);
    for (uint i = 0; i < _num_packet_per_stream; ++i) { // 16 or 80 frames in a stream
        uint64_t fpga_now = (fpga_seq_num + _timesamples_per_pulsar_packet * i);
        psr_header->seconds = time_now->tv_sec - unix_offset;
        psr_header->data_frame =
            (time_now->tv_nsec / 1.e9) / (_timesamples_per_pulsar_packet * fpga_s);

        // Load frequency indices into thread_id and EUD3 if frequency packing.
        psr_header->thread_id = thread_ids[0];
        psr_header->eud3 = (uint32_t)thread_ids[1] + ((uint32_t)thread_ids[2] << 10)
                           + ((uint32_t)thread_ids[3] << 20);

        for (uint32_t beam_id = 0; beam_id < _num_pulsar_beams; ++beam_id) {
            psr_header->eud1 = beam_id; // beam id
            psr_header->eud2 = beam_coord.scaling[beam_id];
            uint16_t ra_part = (uint16_t)(beam_coord.right_ascension[beam_id] * 100);
            uint16_t dec_part = (uint16_t)((beam_coord.declination[beam_id] + 90) * 100);
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
            memcpy(&out_buf[beam_id * _num_packet_per_stream * _udp_pulsar_packet_size
                            + i * _udp_pulsar_packet_size],
                   psr_header, sizeof(PSRHeader));
        }
        // Increment time for the next frame
        time_now->tv_nsec += _timesamples_per_pulsar_packet * fpga_ns;
        if (time_now->tv_nsec > 999999999) {
            time_now->tv_sec += time_now->tv_nsec / 1000000000u;
            time_now->tv_nsec = time_now->tv_nsec % 1000000000u;
        }
    } // end packet
}

void pulsarPostProcess::main_thread() {

    auto& tel = Telescope::instance();
    uint64_t fpga_ns = tel.seq_length_nsec();

    struct timespec time_now = {0, 0};

    int out_buffer_ID = 0;
    int startup = 1; // related to the likely & unlikely

    uint16_t thread_ids[_num_freqs_per_gpu] = {0};

    int in_buffer_ID = 0;
    uint64_t frame_fpga_seq_num = 0;     // sample starting the current input frame
    uint32_t current_input_location = 0; // goes from 0 to _samples_per_data_set

    PSRHeader psr_header;
    psr_header.seconds = 0; // UD
    psr_header.legacy = 0;
    psr_header.invalid = 0;
    psr_header.data_frame = 0; // UD
    psr_header.ref_epoch = 36; // First half of 2018.
    psr_header.unused = 0;
    assert(_timesamples_per_pulsar_packet == 625);
    psr_header.frame_len = 629;  // 5032-B = _timeseamples_per_pulsar_packet * _freqs_per_stream *
                                 // _num_pols * 1-B + 32-B
    psr_header.log_num_chan = 3; // log_2(8), 4 freqs, 2 pols
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

    chordMetadata::beamCoord beam_coord;
    const uint32_t num_out_frames = _num_freqs_per_gpu / num_freqs_per_frame;
    assert(num_out_frames > 0);
    assert((uint32_t)pulsar_buf->num_frames >= num_out_frames);
    assert(num_out_frames * num_freqs_per_frame == _num_freqs_per_gpu);
    uint8_t* out_frames[num_out_frames] = {nullptr};
    // Get the first output buffer which will always be id = 0 to start.
    for (uint32_t freq_block = 0; freq_block < num_out_frames; ++freq_block) {
        uint8_t* out_frame = pulsar_buf->wait_for_empty_frame(
            unique_name, (out_buffer_ID + freq_block) % pulsar_buf->num_frames);
        if (out_frame == nullptr)
            goto end_loop;
        out_frames[freq_block] = out_frame;
    }

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        const uint8_t* in_frame = in_buffer->wait_for_full_frame(unique_name, in_buffer_ID);
        if (in_frame == nullptr)
            return;
        const int64_t new_frame_fpga_seq_num = get_fpga_seq_num(in_buffer, in_buffer_ID);

        // Initialize data for header info, namely position and frequency labels.
        beam_coord = get_beam_coord(in_buffer, in_buffer_ID);
        thread_ids[0] = tel.to_freq_id(in_buffer, in_buffer_ID);
        // TODO: replace by correct frequency ids
        for (unsigned int f = 1; f < _num_freqs_per_gpu; f++)
            thread_ids[f] = thread_ids[0];

#if 0   // station id is strange since more than one F-engine would have contributed anyway, so for
        // now, just make up something
        ice_stream_id_t stream_id = ice_extract_stream_id_t(get_chord_metadata(in_buf[0], in_buffer_ID[0])->get_stream_id());
        psr_header.station_id =
            (uint16_t)(stream_id.crate_id * 16 + stream_id.slot_id + stream_id.link_id * 32);
#endif
        psr_header.station_id = 0;

        bool skipped_frames = (new_frame_fpga_seq_num - frame_fpga_seq_num) > _samples_per_data_set;
        frame_fpga_seq_num = new_frame_fpga_seq_num;

        // If this is the first time wait until we get the start of a being a
        // multiple of the packet length
        if (unlikely(startup == 1)) {
            startup = 0;

            if (!tel.gps_time_enabled()) {
                ERROR("[Time Check] gps global time is not set.");
            }
            time_now = tel.to_time(frame_fpga_seq_num);
            uint32_t pkt_length_in_ns = _timesamples_per_pulsar_packet * fpga_ns;
            uint32_t ns_offset = pkt_length_in_ns - (time_now.tv_nsec % pkt_length_in_ns);
            if (ns_offset == pkt_length_in_ns)
                ns_offset = 0;
            float seq_number_offset_float = ns_offset / (float)fpga_ns;
            uint seq_number_offset = round(seq_number_offset_float);
            assert(seq_number_offset == 0);

            current_input_location = seq_number_offset;
            fpga_seq_num = frame_fpga_seq_num + seq_number_offset;
            time_now.tv_nsec += seq_number_offset * fpga_ns;
            while (time_now.tv_nsec >= 1e9) {
                time_now.tv_sec += 1;
                time_now.tv_nsec -= 1e9;
            }
            skipped_frames = false;

            // Fill the first output buffer headers
            struct timespec time_dummy;
            for (uint32_t freq_block = 0; freq_block < num_out_frames; ++freq_block) {
                time_dummy = time_now; // modified (!) by fill_headers
                assert(freq_block < num_out_frames);
                fill_headers((unsigned char*)out_frames[freq_block], &psr_header, fpga_seq_num,
                             &time_dummy, beam_coord,
                             &thread_ids[freq_block * num_freqs_per_frame]);
            }
            time_now = time_dummy;
        }
        assert(time_now.tv_sec != 0); // initialized (unless we run 1970-Jan-01)

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
                if (ns_offset == pkt_length_in_ns)
                    ns_offset = 0;
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
                struct timespec time_dummy;
                for (uint32_t freq_block = 0; freq_block < num_out_frames; ++freq_block) {
                    time_dummy = time_now; // modified (!) by fill_headers
                    assert(freq_block < num_out_frames);
                    fill_headers((unsigned char*)out_frames[freq_block], &psr_header, fpga_seq_num,
                                 &time_dummy, beam_coord,
                                 &thread_ids[freq_block * num_freqs_per_frame]);
                }
                time_now = time_dummy;
            }

            // now store samples into output buffer.
            assert(_samples_per_data_set * _num_pol * _num_beams * _num_freqs_per_gpu
                   == in_buffer->frame_size);
            for (uint i = current_input_location; i < _samples_per_data_set; ++i) {
                if (in_frame_location == _timesamples_per_pulsar_packet) { // last sample
                    in_frame_location = 0;
                    frame++;
                    if (frame == _num_packet_per_stream) { // last frame
                        frame = 0;
                        for (uint32_t freq_block = 0; freq_block < num_out_frames; ++freq_block) {
                            pulsar_buf->mark_frame_full(unique_name, (out_buffer_ID + freq_block)
                                                                         % pulsar_buf->num_frames);
                            out_frames[freq_block] = nullptr;
                        }
                        // Get a new output buffer
                        out_buffer_ID = (out_buffer_ID + num_out_frames) % pulsar_buf->num_frames;
                        for (uint32_t freq_block = 0; freq_block < num_out_frames; ++freq_block) {
                            uint8_t* out_frame = pulsar_buf->wait_for_empty_frame(
                                unique_name, (out_buffer_ID + freq_block) % pulsar_buf->num_frames);
                            if (out_frame == nullptr)
                                goto end_loop;
                            assert(freq_block < num_out_frames);
                            assert(out_frames[freq_block] == nullptr);
                            out_frames[freq_block] = out_frame;
                        }
                        // Fill the headers of the new buffer
                        fpga_seq_num += _timesamples_per_pulsar_packet * _num_packet_per_stream;
                        struct timespec time_dummy;
                        for (uint32_t freq_block = 0; freq_block < num_out_frames; ++freq_block) {
                            time_dummy = time_now; // modified (!) by fill_headers
                            assert(freq_block < num_out_frames);
                            fill_headers((unsigned char*)out_frames[freq_block], &psr_header,
                                         fpga_seq_num, &time_dummy, beam_coord,
                                         &thread_ids[freq_block * num_freqs_per_frame]);
                        }
                        time_now = time_dummy;
                    } // end if last frame
                } // end if last sample

                for (uint32_t thread_id = 0; thread_id < _num_freqs_per_gpu; ++thread_id) {
                    const uint32_t freq_block = thread_id / num_freqs_per_frame;
                    assert(freq_block < num_out_frames);
                    uint8_t* out_buf = out_frames[freq_block];
                    uint8_t const* const in_buf_data = in_frame;

                    for (uint32_t psr = 0; psr < _num_pulsar_beams; ++psr) { // loop psr
                        for (uint32_t p = 0; p < _num_pol; ++p) {
                            // beam->packets->[freq-time-pol]
                            const uint32_t out_index =
                                // beam
                                psr * _udp_pulsar_packet_size * _num_packet_per_stream
                                // packet
                                + frame * _udp_pulsar_packet_size
                                // freq
                                + (thread_id % num_freqs_per_frame) * _timesamples_per_pulsar_packet
                                      * _num_pol
                                // time
                                + in_frame_location * _num_pol
                                // pol
                                + p
                                // offset for packet header
                                + udp_pulsar_header_size;

                            const uint32_t in_index =
                                psr * _num_freqs_per_gpu * _num_pol * _samples_per_data_set
                                + (thread_id % _num_freqs_per_gpu) * _num_pol
                                      * _samples_per_data_set
                                + p * _samples_per_data_set + i;
                            assert(out_index < pulsar_buf->frame_size);
                            assert(in_index < in_buffer->frame_size);
                            out_buf[out_index] = in_buf_data[in_index];
                        } // end loop pol
                    } // end loop psr
                } // end loop freq
                in_frame_location++;
            } // end loop time
            current_input_location = 0;
        } // end if not start up

        // Release the input buffer
        in_buffer->mark_frame_empty(unique_name, in_buffer_ID);
        in_buffer_ID = (in_buffer_ID + 1) % in_buffer->num_frames;
    } // end stop thread
end_loop:;
}
