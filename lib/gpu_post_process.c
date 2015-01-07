#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

#include "buffers.h"
#include "errors.h"
#include "chrx.h"
#include "output_formating.h"
#include "config.h"
#include "gpu_post_process.h"

void gpu_post_process_thread(void* arg)
{
    struct gpuPostProcessThreadArg * args = (struct gpuPostProcessThreadArg *) arg;

    int in_buffer_ID = -1;
    int out_buffer_ID = 0;
    int frame_number = 0;

    struct Config * config = args->config;

    int useableBufferIDs[config->gpu.num_gpus][1];
    for (int i = 0; i < config->gpu.num_gpus; ++i) {
        useableBufferIDs[i][0] = 0;
    }
    int link_id = 0;

    // Create tcp send buffer
    int num_values = ((config->processing.num_elements *
    (config->processing.num_elements + 1)) / 2 ) *
    config->processing.num_total_freq;
    int buffer_size = sizeof(struct tcp_frame_header) +
    num_values * (sizeof(complex_int_t) + sizeof(uint8_t));

    int num_vis = ((config->processing.num_elements * (config->processing.num_elements + 1)) / 2 );
    int num_values_per_link = num_vis * config->processing.num_local_freq;

    struct stream_id local_stream_ids[MAX_NUM_LINKS];

    unsigned char * buf = malloc(buffer_size);
    CHECK_MEM(buf);

    unsigned char * data_sets_buf =
    malloc(num_values * config->processing.num_data_sets * sizeof(complex_int_t));
    CHECK_MEM(data_sets_buf);

    // Create convenient pointers into the buffer (yay pointer math).
    struct tcp_frame_header * header = (struct tcp_frame_header *)buf;
    complex_int_t * visibilities = ( complex_int_t * ) (buf + sizeof(struct tcp_frame_header));
    uint8_t * error_flags = (uint8_t *)(buf + sizeof(struct tcp_frame_header)
    + num_values*sizeof(complex_int_t) );
    // Connect to server.
    struct sockaddr_in ch_acq_addr;

    // Wait for full buffers.
    for (;;) {

        int gpu_id = config->fpga_network.link_map[link_id].gpu_id;

        // This call is blocking!
        in_buffer_ID = get_full_buffer_from_list(&args->in_buf[gpu_id], useableBufferIDs[gpu_id], 1);

        // Check if the producer has finished, and we should exit.
        if (in_buffer_ID == -1) {
            mark_producer_done(args->out_buf, 0);
            INFO("Closing gpu_post_process");
            int ret;
            pthread_exit((void *) &ret);
        }

        // TODO Check that this is valid.  Make sure all seq numbers are the same for a frame, etc.
        uint32_t fpga_seq_number = get_fpga_seq_num(&args->in_buf[gpu_id], in_buffer_ID);
        struct timeval frame_start_time = get_first_packet_recv_time(&args->in_buf[gpu_id], in_buffer_ID);

        uint32_t packed_stream_ID = get_streamID(&args->in_buf[gpu_id], in_buffer_ID);
        local_stream_ids[link_id].link_id =   packed_stream_ID & 0x000F;
        local_stream_ids[link_id].slot_id =  (packed_stream_ID & 0x00F0) >> 4;
        local_stream_ids[link_id].crate_id = (packed_stream_ID & 0x0F00) >> 8;
        local_stream_ids[link_id].reserved = (packed_stream_ID & 0xF000) >> 12;

        for (int i = 0; i < config->processing.num_data_sets; ++i) {

            if (config->processing.num_elements <= 16) {
                // TODO Make this cleaner (single function)
                reorganize_32_to_16_element_GPU_correlated_data_with_shuffle(
                    config->processing.num_local_freq,
                    config->processing.num_elements,
                    1,
                    (int *)&args->in_buf[gpu_id].data[in_buffer_ID][i * (args->in_buf[gpu_id].buffer_size / config->processing.num_data_sets)],
                                                                             args->config->processing.product_remap);


                shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8(
                    config->processing.num_local_freq,
                    (int *)&args->in_buf[gpu_id].data[in_buffer_ID][i * (args->in_buf[gpu_id].buffer_size / config->processing.num_data_sets)],
                                                                                                  (complex_int_t *)&data_sets_buf[i * num_values * sizeof(complex_int_t)], link_id );
            } else {
                reorganize_GPU_to_upper_triangle_remap(config->gpu.block_size,
                    config->processing.num_blocks,
                    config->processing.num_local_freq,
                    config->processing.num_elements,
                    1,
                    (int *)&args->in_buf[gpu_id].data[in_buffer_ID][i * (args->in_buf[gpu_id].buffer_size / config->processing.num_data_sets)],
                    (complex_int_t *)&data_sets_buf[(i * num_values + link_id * num_values_per_link) * sizeof(complex_int_t)],
                    config->processing.product_remap);
            }
        }

        // TODO Add flagging code here.
        for (int i = 0; i < num_values; ++i) {
            error_flags[i] = 0;
        }

        if (link_id + 1 == config->fpga_network.num_links) {

            for (int i = 0; i < config->processing.num_data_sets; ++i) {

                // If this is the first frame, set the header, and initial visibility data.
                if (frame_number == 0) {
                    header->cpu_timestamp = frame_start_time;
                    double time_offset = i * (config->processing.samples_per_data_set * 2.56);
                    header->cpu_timestamp.tv_usec = header->cpu_timestamp.tv_usec + time_offset;
                    header->fpga_seq_number = fpga_seq_number*config->fpga_network.timesamples_per_packet + i * config->processing.samples_per_data_set;
                    header->num_freq = config->processing.num_total_freq;
                    header->num_vis = num_vis;
                    for (int j = 0; j < MAX_NUM_LINKS; ++j) {
                        header->stream_ids[j] = local_stream_ids[j];
                    }
                    for (int j = 0; j < num_values; ++j) {
                        visibilities[j] = *(complex_int_t *)(data_sets_buf + i * (num_values * sizeof(complex_int_t)) + j * sizeof(complex_int_t));
                    }
                } else {
                    // Add to the visibilities.
                    for (int j = 0; j < num_values; ++j) {
                        complex_int_t temp_vis = *(complex_int_t *)(data_sets_buf + i * (num_values * sizeof(complex_int_t)) + j * sizeof(complex_int_t));
                        visibilities[j].real += temp_vis.real;
                        visibilities[j].imag += temp_vis.imag;
                    }
                }

                // If we are on the last frame in the set, push the buffer to the network thread.
                if (frame_number + 1 >= config->processing.num_gpu_frames) {
                    INFO("Sending TCP frame to network thread: FPGA_SEQ_NUMBER = %u ; NUM_FREQ = %d ; NUM_VIS = %d ; BUFFER_SIZE = %d",
                         header->fpga_seq_number*config->fpga_network.timesamples_per_packet,
                         header->num_freq, header->num_vis, buffer_size);

                    wait_for_empty_buffer(args->out_buf, out_buffer_ID);

                    memcpy(args->out_buf->data[out_buffer_ID], buf, buffer_size);
                    mark_buffer_full(args->out_buf, out_buffer_ID);
                    out_buffer_ID = (out_buffer_ID + 1) % args->out_buf->num_buffers;
                }
            }

            frame_number = (frame_number + 1) % config->processing.num_gpu_frames;
        }

        release_info_object(&args->in_buf[gpu_id], in_buffer_ID);
        mark_buffer_empty(&args->in_buf[gpu_id], in_buffer_ID);

        useableBufferIDs[gpu_id][0] = (useableBufferIDs[gpu_id][0] + 1) % args->in_buf->num_buffers;

        link_id = (link_id + 1) % config->fpga_network.num_links;
    }

}
