#include "beamformingPostProcess.hpp"

#include "BranchPrediction.hpp" // for unlikely, likely
#include "Config.hpp"           // for Config
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"        // for stream_t
#include "buffer.hpp"           // for Buffer, wait_for_empty_frame, mark_frame_empty, mark_fra...
#include "bufferContainer.hpp"  // for bufferContainer
#include "chimeMetadata.hpp"    // for get_fpga_seq_num, get_first_packet_recv_time, get_stream_id
#include "vdif_functions.h"     // for VDIFHeader

#include "fmt.hpp" // for format, fmt

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <math.h>     // for round
#include <regex>      // for match_results<>::_Base_type
#include <stdlib.h>   // for free, malloc
#include <string.h>   // for memcpy
#include <string>     // for string
#include <sys/time.h> // for timeval

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(beamformingPostProcess);

beamformingPostProcess::beamformingPostProcess(Config& config, const std::string& unique_name,
                                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&beamformingPostProcess::main_thread, this)) {
    _num_fpga_links = config.get<uint32_t>(unique_name, "num_links");
    //_num_gpus = config.get_int("/gpu", "num_gpus");
    _num_gpus = config.get<uint32_t>(unique_name, "num_gpus");
    in_buf = (Buffer**)malloc(_num_gpus * sizeof(Buffer*));
    for (uint32_t i = 0; i < _num_gpus; ++i) {
        in_buf[i] = get_buffer(fmt::format(fmt("beam_in_buf_{:d}"), i));
        in_buf[i]->register_consumer(unique_name);
    }
    vdif_buf = get_buffer("vdif_out_buf");
    vdif_buf->register_producer(unique_name);
}

beamformingPostProcess::~beamformingPostProcess() {
    free(in_buf);
}

void beamformingPostProcess::fill_headers(unsigned char* out_buf, struct VDIFHeader* vdif_header,
                                          const uint32_t second, const uint32_t fpga_seq_num,
                                          const uint32_t num_links, uint32_t* thread_id) {
    // Populate the headers
    vdif_header->seconds = second;
    assert(sizeof(struct VDIFHeader) == 32);

    for (int i = 0; i < 625; ++i) {
        vdif_header->data_frame = i;

        for (uint32_t j = 0; j < num_links; ++j) {
            vdif_header->thread_id = thread_id[j];
            vdif_header->eud2 = fpga_seq_num + 625 * i;

            // Each polarization is its own station
            vdif_header->station_id = 0;
            memcpy(&out_buf[(i * 16 + j * 2) * 5032], vdif_header, sizeof(struct VDIFHeader));
            vdif_header->station_id = 1;
            memcpy(&out_buf[(i * 16 + j * 2 + 1) * 5032], vdif_header, sizeof(struct VDIFHeader));
        }
    }
}

void beamformingPostProcess::main_thread() {

    // Apply config.
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<uint32_t>(unique_name, "num_data_sets");
    _link_map = config.get<std::vector<int32_t>>(unique_name, "link_map");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");

    int in_buffer_ID[_num_gpus];
    int in_buffer_ID_final[_num_gpus];
    uint8_t* in_frame[_num_fpga_links];
    int out_buffer_ID = 0;
    int startup = 1;

    for (uint32_t i = 0; i < _num_gpus; ++i) {
        in_buffer_ID[i] = 0;
        in_buffer_ID_final[i] = 0;
    }

    const uint32_t num_samples = _samples_per_data_set * _num_data_sets;
    uint32_t current_input_location = 0;

    // Constants to speed things up
    const uint32_t frame_size = 5032; // TODO don't make this constant.
    const uint32_t header_size = 32;

    uint32_t thread_ids[_num_fpga_links];

    // Header template
    struct VDIFHeader vdif_header;
    vdif_header.bits_depth = 3;  // 4-bit data
    vdif_header.data_type = 1;   // Complex
    vdif_header.frame_len = 629; // 5032 bytes / 8
    vdif_header.invalid = 0;
    vdif_header.legacy = 0;
    vdif_header.log_num_chan = 3; // log(8)/log(2)
    vdif_header.ref_epoch = 0;    // TODO set this dynamically from the current date.
    vdif_header.unused = 0;
    vdif_header.vdif_version = 1; // Correct?
    vdif_header.edv = 0;
    vdif_header.eud1 = 0;
    vdif_header.eud2 = 0;
    vdif_header.eud3 = 0;
    vdif_header.eud4 = 0;

    int frame = 0;
    int in_frame_location = 0;
    int second = 0;
    uint32_t fpga_seq_num = 0;

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* vdif_frame = vdif_buf->wait_for_empty_frame(unique_name, out_buffer_ID);

    while (!stop_thread) {

        // INFO("beamforming_post_process; waiting for GPU output.");

        uint32_t first_seq_number = 0;

        // Get all the input buffers needed to form the output.
        for (uint32_t i = 0; i < _num_fpga_links; ++i) {

            // Get an input buffer
            int gpu_id = _link_map[i];

            // This call is blocking!
            in_frame[i] =
                in_buf[gpu_id]->wait_for_full_frame(unique_name, in_buffer_ID[gpu_id]);
            if (in_frame[i] == nullptr)
                goto end_loop;

            if (i == 0) {
                first_seq_number =
                    (uint32_t)get_fpga_seq_num(in_buf[_link_map[0]], in_buffer_ID[0]);
            } else {
                assert(first_seq_number
                       == (uint32_t)get_fpga_seq_num(in_buf[gpu_id], in_buffer_ID[gpu_id]));
            }

            // TODO: port this to use ice_extract_stream_id_t
            stream_t stream_id = get_stream_id(in_buf[gpu_id], in_buffer_ID[gpu_id]);
            int link_id = stream_id.id & 0x000F;
            int slot_id = (stream_id.id & 0x00F0) >> 4;
            thread_ids[i] = link_id + (slot_id << 4);

            in_buffer_ID[gpu_id] = (in_buffer_ID[gpu_id] + 1) % in_buf[gpu_id]->num_frames;
        }

        // INFO("beamforming_post_process; got full set of GPU output buffers");

        // If this is the first time wait until we get the start of an interger second period.
        if (unlikely(startup == 1)) {

            // testing sync code
            startup = 0;
            current_input_location = 0;
            struct timeval time = get_first_packet_recv_time(in_buf[_link_map[0]], 0);
            second = (int)(round((double)time.tv_sec / 20.0) * 20.0) - 946728000;
            // Fill the first output buffer headers
            fpga_seq_num = first_seq_number;
            fill_headers((unsigned char*)vdif_frame, &vdif_header, second, first_seq_number,
                         _num_fpga_links, thread_ids);
            // second = get_vdif_second(first_seq_number + current_input_location);
        }

        // This loop which takes data from the input buffer and formats the output.
        if (likely(startup == 0)) {

            // frame = get_vdif_frame( first_seq_number + current_input_location );
            // in_frame_location = get_vdif_location( first_seq_number + current_input_location );

            for (uint32_t i = current_input_location; i < num_samples; ++i) {

                if (unlikely(in_frame_location == 625)) {
                    in_frame_location = 0;
                    frame++;
                    if (unlikely(frame == 625)) {
                        frame = 0;
                        second++;

                        vdif_buf->mark_frame_full(unique_name, out_buffer_ID);

                        // Get a new output buffer
                        out_buffer_ID = (out_buffer_ID + 1) % vdif_buf->num_frames;
                        vdif_frame =
                            vdif_buf->wait_for_empty_frame(unique_name, out_buffer_ID);
                        if (vdif_frame == nullptr)
                            goto end_loop;

                        // Fill the headers of the new buffer
                        fpga_seq_num += 625 * 625;
                        fill_headers((unsigned char*)vdif_frame, &vdif_header, second, fpga_seq_num,
                                     _num_fpga_links, thread_ids);
                    }
                }

                for (uint32_t thread_id = 0; thread_id < _num_fpga_links; ++thread_id) {
                    unsigned char* out_buf = (unsigned char*)vdif_frame;
                    uint32_t station_0_index = frame * frame_size * _num_fpga_links * 2
                                               + thread_id * frame_size * 2 + in_frame_location * 8
                                               + header_size;

                    uint32_t station_1_index = frame * frame_size * _num_fpga_links * 2
                                               + thread_id * frame_size * 2 + frame_size
                                               + in_frame_location * 8 + header_size;

                    // DEBUG("beamforming_post_process: station_0_index = {:d}", station_0_index);

                    for (uint32_t freq = 0; freq < _num_local_freq; ++freq) {
                        unsigned char* in_buf_data = (unsigned char*)in_frame[thread_id];
                        // The two polarizations.
                        // Each sample is 4-bit real, 4-bit imaginary, so byte operations work just
                        // fine here.
                        out_buf[station_0_index + freq] = in_buf_data[i * 16 + freq * 2];
                        out_buf[station_1_index + freq] = in_buf_data[i * 16 + freq * 2 + 1];
                    }
                }

                in_frame_location++;
            }
            current_input_location = 0;
        }

        // Release the input buffers
        for (uint32_t i = 0; i < _num_fpga_links; ++i) {
            int gpu_id = _link_map[i];

            in_buf[gpu_id]->mark_frame_empty(unique_name, in_buffer_ID_final[gpu_id]);
            in_buffer_ID_final[gpu_id] =
                (in_buffer_ID_final[gpu_id] + 1) % in_buf[gpu_id]->num_frames;
        }
    }
end_loop:;
}
