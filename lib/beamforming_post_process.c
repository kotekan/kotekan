#include <assert.h>
#include <stdlib.h>
#include <string.h>

// TODO Where do these live?
# define likely(x)      __builtin_expect(!!(x), 1)
# define unlikely(x)    __builtin_expect(!!(x), 0)

#include "beamforming_post_process.h"
#include "util.h"
#include "errors.h"
#include "time_tracking.h"

void fill_headers(unsigned char * out_buf,
                  struct VDIFHeader * vdif_header,
                  const uint32_t second,
                  const uint32_t num_links,
                  uint32_t *thread_id) {
    // Populate the headers
    vdif_header->seconds = second;
    assert(sizeof(struct VDIFHeader) == 32);

    for (int i = 0; i < 625; ++i) {
        vdif_header->data_frame = i;

        for(int j = 0; j < num_links; ++j) {
            vdif_header->therad_id = thread_id[j];
            // Each polarization is its own station
            vdif_header->station_id = 0;
            memcpy(&out_buf[(i*16+j*2)*5032], vdif_header, sizeof(struct VDIFHeader));
            vdif_header->station_id = 1;
            memcpy(&out_buf[(i*16+j*2 + 1)*5032], vdif_header, sizeof(struct VDIFHeader));
        }
    }
}

void beamforming_post_process(void* arg)
{
    struct BeamformingPostProcessArgs * args = (struct BeamformingPostProcessArgs *) arg;

    struct Config * config = args->config;

    int in_buffer_ID[config->fpga_network.num_links];
    int out_buffer_ID = 0;
    int startup = 1;

    int useableBufferIDs[config->gpu.num_gpus][1];
    for (int i = 0; i < config->gpu.num_gpus; ++i) {
        useableBufferIDs[i][0] = 0;
    }

    const uint32_t num_samples = config->processing.samples_per_data_set * config->processing.num_data_sets;
    uint32_t current_input_location = 0;

    // Constants to speed things up
    const uint32_t num_links = config->fpga_network.num_links;
    const uint32_t frame_size = 5032; // TODO don't make this constant.
    const uint32_t header_size = 32;

    uint32_t thread_ids[num_links];

    // Header template
    struct VDIFHeader vdif_header;
    vdif_header.bits_depth = 4; // 4-bit data
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

    // Get the first output buffer which will always be id = 0 to start.
    wait_for_empty_buffer(args->out_buf, out_buffer_ID);

    for(EVER) {

        //INFO("beamforming_post_process; waiting for GPU output.");

        // Get all the input buffers needed to form the output.
        for (int i = 0; i < num_links; ++i) {
            // Get an input buffer
            int gpu_id = config->fpga_network.link_map[i].gpu_id;

            // This call is blocking!
            in_buffer_ID[i] = get_full_buffer_from_list(&args->in_buf[gpu_id], useableBufferIDs[gpu_id], 1);

            // Check if the producer has finished, and we should exit.
            if (in_buffer_ID[i] == -1) {
                mark_producer_done(args->out_buf, 0);
                INFO("Closing beamforming_post_process");
                int ret;
                pthread_exit((void *) &ret);
            }

            useableBufferIDs[gpu_id][0] = (useableBufferIDs[gpu_id][0] + 1) % args->in_buf[gpu_id].num_buffers;
        }

        //INFO("beamforming_post_process; got full set of GPU output buffers");

        uint32_t first_seq_number =
            get_fpga_seq_num(&args->in_buf[config->fpga_network.link_map[0].gpu_id], in_buffer_ID[0])
            * config->fpga_network.timesamples_per_packet;

        for (int i = 0; i < num_links; ++i) {
            int gpu_id = config->fpga_network.link_map[i].gpu_id;
            //INFO("fpga_seq_num[%d] = %u", i, get_fpga_seq_num(&args->in_buf[gpu_id], in_buffer_ID[i])
            //        * config->fpga_network.timesamples_per_packet);
            assert(first_seq_number ==
                    get_fpga_seq_num(&args->in_buf[gpu_id], in_buffer_ID[i])
                    * config->fpga_network.timesamples_per_packet);

            int stream_id = get_streamID(&args->in_buf[gpu_id], in_buffer_ID[i]);
            int link_id = stream_id & 0x000F;
            int slot_id = (stream_id & 0x00F0) >> 4;
            thread_ids[i] = link_id + (slot_id << 4);
        }

        // If this is the first time wait until we get the start of an interger second period.
        if (unlikely(startup == 1)) {
            // TODO This can be done without a loop!
            for (int i = 0; i < num_samples; ++i) {
                if (get_vdif_frame( first_seq_number + i ) == 0 &&
                    get_vdif_location( first_seq_number + i ) == 0) {

                    current_input_location = i;
                    startup = 0;
                    break;
                }
            }
            // Fill the first output buffer headers
            fill_headers(args->out_buf->data[out_buffer_ID],
                         &vdif_header,
                         get_vdif_second(first_seq_number + current_input_location),
                         num_links,
                         thread_ids);
        }

        // This loop which takes data from the input buffer and formats the output.
        if (likely(startup == 0)) {

            int frame = get_vdif_frame( first_seq_number + current_input_location );
            int in_frame_location = get_vdif_location( first_seq_number + current_input_location );

            for (int i = current_input_location; i < num_samples; ++i) {

                if (unlikely(in_frame_location == 625)) {
                    in_frame_location = 0;
                    frame++;
                    if (unlikely(frame == 625)) {
                        frame = 0;
                        // The current buffer is full.
                        mark_buffer_full(args->out_buf, out_buffer_ID);

                        // Get a new output buffer
                        out_buffer_ID = (out_buffer_ID + 1) % args->out_buf->num_buffers;
                        wait_for_empty_buffer(args->out_buf, out_buffer_ID);

                        // Fill the headers of the new buffer
                        fill_headers(args->out_buf->data[out_buffer_ID],
                                     &vdif_header,
                                     get_vdif_second(first_seq_number),
                                     num_links,
                                     thread_ids);
                    }
                }

                for (int thread_id = 0; thread_id < config->fpga_network.num_links; ++thread_id) {
                    unsigned char * out_buf = args->out_buf->data[out_buffer_ID];
                    uint32_t station_0_index = frame * frame_size * num_links * 2
                                                + thread_id * frame_size * 2
                                                + in_frame_location * 8 + header_size;

                    uint32_t station_1_index = frame * frame_size * num_links * 2
                                                + thread_id * frame_size * 2 + frame_size
                                                + in_frame_location * 8 + header_size;

                    for (int freq = 0; freq < config->processing.num_local_freq; ++freq) {
                        int gpu_id = config->fpga_network.link_map[thread_id].gpu_id;
                        unsigned char * in_buf = args->in_buf[gpu_id].data[in_buffer_ID[thread_id]];
                        // The two polarizations.
                        // Each sample is 4-bit real, 4-bit complex, so byte operations work just fine here.
                        out_buf[station_0_index + freq] = in_buf[i*16 + freq*2];
                        out_buf[station_1_index + freq] = in_buf[i*16 + freq*2 + 1];
                    }
                }

                in_frame_location++;
            }
            current_input_location = 0;
        }

        // Release the input buffers
        for (int i = 0; i < num_links; ++i) {
            int gpu_id = config->fpga_network.link_map[i].gpu_id;

            release_info_object(&args->in_buf[gpu_id], in_buffer_ID[i]);
            mark_buffer_empty(&args->in_buf[gpu_id], in_buffer_ID[i]);
        }
    }
}