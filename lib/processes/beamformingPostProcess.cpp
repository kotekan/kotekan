#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <functional>
#include <string>

using std::string;

// TODO Where do these live?
# define likely(x)      __builtin_expect(!!(x), 1)
# define unlikely(x)    __builtin_expect(!!(x), 0)

#include "beamformingPostProcess.hpp"
#include "util.h"
#include "errors.h"
#include "time_tracking.h"
#include "vdif_functions.h"

beamformingPostProcess::beamformingPostProcess(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&beamformingPostProcess::main_thread, this))
{
    _num_fpga_links = config.get_int("/dpdk/num_links");
    _num_gpus = config.get_int("/gpu/num_gpus");
    in_buf = (struct Buffer **)malloc(_num_gpus * sizeof (struct Buffer *));
    for (int i = 0; i < _num_gpus; ++i) {
        in_buf[i] = buffer_container.get_buffer("beamforming_buf" + std::to_string(i));
    }
    vdif_buf = buffer_container.get_buffer("beamform_vdif_buf");

}

beamformingPostProcess::~beamformingPostProcess() {
    free(in_buf);
}

void beamformingPostProcess::fill_headers(unsigned char * out_buf,
                  struct VDIFHeader * vdif_header,
                  const uint32_t second,
                  const uint32_t fpga_seq_num,
                  const uint32_t num_links,
                  uint32_t *thread_id) {
    // Populate the headers
    vdif_header->seconds = second;
    assert(sizeof(struct VDIFHeader) == 32);

    for (int i = 0; i < 625; ++i) {
        vdif_header->data_frame = i;

        for(int j = 0; j < num_links; ++j) {
            vdif_header->thread_id = thread_id[j];
            vdif_header->eud2 = fpga_seq_num + 625 * i;

            // Each polarization is its own station
            vdif_header->station_id = 0;
            memcpy(&out_buf[(i*16+j*2)*5032], vdif_header, sizeof(struct VDIFHeader));
            vdif_header->station_id = 1;
            memcpy(&out_buf[(i*16+j*2 + 1)*5032], vdif_header, sizeof(struct VDIFHeader));
        }
    }
}

void beamformingPostProcess::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    _samples_per_data_set = config.get_int("/processing/samples_per_data_set");
    _num_data_sets = config.get_int("/processing/num_data_sets");
    _link_map = config.get_int_array("/gpu/link_map");
    _num_local_freq = config.get_int("/processing/num_local_freq");
}

void beamformingPostProcess::main_thread() {

    apply_config(0);

    int in_buffer_ID[_num_fpga_links];
    int out_buffer_ID = 0;
    int startup = 1;

    int useableBufferIDs[_num_gpus][1];
    for (int i = 0; i < _num_gpus; ++i) {
        useableBufferIDs[i][0] = 0;
    }

    const uint32_t num_samples = _samples_per_data_set * _num_data_sets;
    uint32_t current_input_location = 0;

    // Constants to speed things up
    const uint32_t frame_size = 5032; // TODO don't make this constant.
    const uint32_t header_size = 32;

    uint32_t thread_ids[_num_fpga_links];

    // Header template
    struct VDIFHeader vdif_header;
    vdif_header.bits_depth = 3; // 4-bit data
    vdif_header.data_type = 1; // Complex
    vdif_header.frame_len = 629; // 5032 bytes / 8
    vdif_header.invalid = 0;
    vdif_header.legacy = 0;
    vdif_header.log_num_chan = 3; // log(8)/log(2)
    vdif_header.ref_epoch = 0; // TODO set this dynamically from the current date.
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
    wait_for_empty_buffer(vdif_buf, out_buffer_ID);

    for(EVER) {

        //INFO("beamforming_post_process; waiting for GPU output.");

        // Get all the input buffers needed to form the output.
        for (uint32_t i = 0; i < _num_fpga_links; ++i) {
            // Get an input buffer
            int gpu_id = _link_map[i];

            // This call is blocking!
            in_buffer_ID[i] = get_full_buffer_from_list(in_buf[gpu_id], useableBufferIDs[gpu_id], 1);

            // Check if the producer has finished, and we should exit.
            if (in_buffer_ID[i] == -1) {
                mark_producer_done(vdif_buf, 0);
                INFO("Closing beamforming_post_process");
                int ret;
                pthread_exit((void *) &ret);
            }

            useableBufferIDs[gpu_id][0] = (useableBufferIDs[gpu_id][0] + 1) % in_buf[gpu_id]->num_buffers;
        }

        //INFO("beamforming_post_process; got full set of GPU output buffers");

        uint32_t first_seq_number =
            (uint32_t)get_fpga_seq_num(in_buf[_link_map[0]], in_buffer_ID[0]);

        for (uint32_t i = 0; i < _num_fpga_links; ++i) {
            int gpu_id = _link_map[i];

            assert(first_seq_number ==
                    (uint32_t)get_fpga_seq_num(in_buf[gpu_id], in_buffer_ID[i]));

            int stream_id = get_streamID(in_buf[gpu_id], in_buffer_ID[i]);
            int link_id = stream_id & 0x000F;
            int slot_id = (stream_id & 0x00F0) >> 4;
            thread_ids[i] = link_id + (slot_id << 4);
        }

        // If this is the first time wait until we get the start of an interger second period.
        if (unlikely(startup == 1)) {

            // testing sync code
            startup = 0;
            current_input_location = 0;
            struct timeval time = get_first_packet_recv_time(in_buf[_link_map[0]], in_buffer_ID[0]);
            second = (int)(round((double)time.tv_sec / 20.0) * 20.0) - 946728000;
            // Fill the first output buffer headers
            fpga_seq_num = first_seq_number;
            fill_headers(vdif_buf->data[out_buffer_ID],
                         &vdif_header,
                         second,
                         first_seq_number,
                         _num_fpga_links,
                         thread_ids);
            //second = get_vdif_second(first_seq_number + current_input_location);
        }

        // This loop which takes data from the input buffer and formats the output.
        if (likely(startup == 0)) {

            // frame = get_vdif_frame( first_seq_number + current_input_location );
            // in_frame_location = get_vdif_location( first_seq_number + current_input_location );

            for (int i = current_input_location; i < num_samples; ++i) {

                if (unlikely(in_frame_location == 625)) {
                    in_frame_location = 0;
                    frame++;
                    if (unlikely(frame == 625)) {
                        frame = 0;
                        second++;

                        mark_buffer_full(vdif_buf, out_buffer_ID);

                        // Get a new output buffer
                        out_buffer_ID = (out_buffer_ID + 1) % vdif_buf->num_buffers;
                        wait_for_empty_buffer(vdif_buf, out_buffer_ID);

                        // Fill the headers of the new buffer
                        fpga_seq_num += 625*625;
                        fill_headers(vdif_buf->data[out_buffer_ID],
                                     &vdif_header,
                                     second,
                                     fpga_seq_num,
                                     _num_fpga_links,
                                     thread_ids);
                    }
                }

                for (int thread_id = 0; thread_id < _num_fpga_links; ++thread_id) {
                    unsigned char * out_buf = vdif_buf->data[out_buffer_ID];
                    uint32_t station_0_index = frame * frame_size * _num_fpga_links * 2
                                                + thread_id * frame_size * 2
                                                + in_frame_location * 8 + header_size;

                    uint32_t station_1_index = frame * frame_size * _num_fpga_links * 2
                                                + thread_id * frame_size * 2 + frame_size
                                                + in_frame_location * 8 + header_size;

                    //DEBUG("beamforming_post_process: station_0_index = %d", station_0_index);

                    for (int freq = 0; freq < _num_local_freq; ++freq) {
                        int gpu_id = _link_map[thread_id];
                        unsigned char * in_buf_data = in_buf[gpu_id]->data[in_buffer_ID[thread_id]];
                        // The two polarizations.
                        // Each sample is 4-bit real, 4-bit complex, so byte operations work just fine here.
                        out_buf[station_0_index + freq] = in_buf_data[i*16 + freq*2];
                        out_buf[station_1_index + freq] = in_buf_data[i*16 + freq*2 + 1];
                    }
                }

                in_frame_location++;
            }
            current_input_location = 0;
        }

        // Release the input buffers
        for (int i = 0; i < _num_fpga_links; ++i) {
            int gpu_id = _link_map[i];

            release_info_object(in_buf[gpu_id], in_buffer_ID[i]);
            mark_buffer_empty(in_buf[gpu_id], in_buffer_ID[i]);
        }
    }
}
