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

#include "ch_acq_uplink.h"
#include "buffers.h"
#include "errors.h"
#include "chrx.h"
#include "output_formating.h"
#include "config.h"

// A TCP frame contains this header followed by the visibilities, and flags.
// -- HEADER:sizeof(TCP_frame_header) --
// -- VISIBILITIES:n_corr * n_freq * sizeof(complex_int_t) --
// -- FLAGS:n_corr * sizeof(uint8_t) --
#pragma pack(1)
struct tcp_frame_header {
    uint32_t fpga_seq_number;
    uint32_t num_freq;
    uint32_t num_vis; // The number of visibilities per frequency.

    struct timeval cpu_timestamp; // The time stamp as set by the GPU correlator - not accurate!
};
#pragma pack(0)


void ch_acq_uplink_thread(void* arg)
{
    struct ch_acqUplinkThreadArg * args = (struct ch_acqUplinkThreadArg *) arg;

    int bufferID = -1;
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

    int tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd == -1) {
        ERROR("Could not create socket, errno: %d", errno);
        return;
    }

    bzero(&ch_acq_addr, sizeof(ch_acq_addr));
    ch_acq_addr.sin_family = AF_INET;
    ch_acq_addr.sin_addr.s_addr = inet_addr(config->ch_master_network.collection_server_ip);
    ch_acq_addr.sin_port = htons(config->ch_master_network.collection_server_port);

    if (connect(tcp_fd, (struct sockaddr *)&ch_acq_addr, sizeof(ch_acq_addr)) == -1) {
        ERROR("Could not connect to ch_acq, error: %d", errno);
    }

    // Wait for, and transmit, full buffers.
    for (;;) {

        int gpu_id = config->fpga_network.link_map[link_id].gpu_id;

        // This call is blocking!
        bufferID = get_full_buffer_from_list(&args->buf[gpu_id], useableBufferIDs[gpu_id], 1);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            INFO("Closing ch_acq_uplink");
            close(tcp_fd);
            int ret;
            pthread_exit((void *) &ret);
        }

        // TODO Check that this is valid.  Make sure all seq numbers are the same for a frame, etc.
        uint32_t fpga_seq_number = get_fpga_seq_num(&args->buf[gpu_id], bufferID);
        struct timeval frame_start_time = get_first_packet_recv_time(&args->buf[gpu_id], bufferID);

        for (int i = 0; i < config->processing.num_data_sets; ++i) {

            if (config->processing.num_elements <= 16) {
                // TODO Make this cleaner (single function)
                reorganize_32_to_16_element_GPU_correlated_data_with_shuffle(
                    config->processing.num_local_freq,
                    config->processing.num_elements,
                    1,
                    (int *)&args->buf[gpu_id].data[bufferID][i * (args->buf[gpu_id].buffer_size / config->processing.num_data_sets)],
                    args->config->processing.product_remap);


                shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8(
                    config->processing.num_local_freq,
                    (int *)&args->buf[gpu_id].data[bufferID][i * (args->buf[gpu_id].buffer_size / config->processing.num_data_sets)],
                    (complex_int_t *)&data_sets_buf[i * num_values * sizeof(complex_int_t)], link_id );
            } else {
                reorganize_GPU_to_upper_triangle(config->gpu.block_size,
                    config->processing.num_blocks,
                    config->processing.num_local_freq,
                    config->processing.num_elements,
                    1,
                    (int *)&args->buf[gpu_id].data[bufferID][i * (args->buf[gpu_id].buffer_size / config->processing.num_data_sets)],
                    (complex_int_t *)&data_sets_buf[(i * num_values + link_id * num_values_per_link) * sizeof(complex_int_t)]);
            }
        }
        // TODO Add the ability to integrate the data down further here.

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

                // If we are on the last frame in the set, push the buffer.
                if (frame_number + 1 >= config->processing.num_gpu_frames) {
                    INFO("Sending frame to ch_master: FPGA_SEQ_NUMBER = %u ; NUM_FREQ = %d ; NUM_VIS = %d",
                        header->fpga_seq_number*config->fpga_network.timesamples_per_packet,
                        header->num_freq, header->num_vis);

                    ssize_t bytes_sent = send(tcp_fd, buf, buffer_size, 0);
                    if (bytes_sent <= 0) {
                        ERROR("Could not send frame to ch_acq, error: %d", errno);
                        break;
                    }
                    if (bytes_sent != buffer_size) {
                        ERROR("Could not send all bytes: bytes sent = %d; buffer_size = %d", (int)bytes_sent, buffer_size);
                        break;
                    }
                }
            }

            frame_number = (frame_number + 1) % config->processing.num_gpu_frames;
        }

        release_info_object(&args->buf[gpu_id], bufferID);
        mark_buffer_empty(&args->buf[gpu_id], bufferID);

        useableBufferIDs[gpu_id][0] = (useableBufferIDs[gpu_id][0] + 1) % args->buf->num_buffers;

        link_id = (link_id + 1) % config->fpga_network.num_links;
    }

}
