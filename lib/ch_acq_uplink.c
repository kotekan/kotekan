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
    assert(args->num_links > 0);

    int useableBufferIDs[1] = {0};
    int link_id = 0;

    // Create tcp send buffer
    int num_values = ((args->actual_num_elements * (args->actual_num_elements + 1)) / 2 ) * args->total_num_freq;
    int buffer_size = sizeof(struct tcp_frame_header) + num_values * (sizeof(complex_int_t) + sizeof(uint8_t));

    unsigned char * buf = malloc(buffer_size);
    CHECK_MEM(buf);

    unsigned char * data_sets_buf = malloc(num_values * args->num_data_sets * sizeof(complex_int_t));
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
    ch_acq_addr.sin_addr.s_addr = inet_addr(args->ch_acq_ip_addr);
    ch_acq_addr.sin_port = htons(args->ch_acq_port_num);

    if (connect(tcp_fd, (struct sockaddr *)&ch_acq_addr, sizeof(ch_acq_addr)) == -1) {
        ERROR("Could not connect to ch_acq, error: %d", errno);
    }

    // Wait for, and transmit, full buffers.
    for (;;) {

        // This call is blocking!
        bufferID = get_full_buffer_from_list(args->buf, useableBufferIDs, 1);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            int ret;
            pthread_exit((void *) &ret);
        }

        // TODO Check that this is valid.  Make sure all seq numbers are the same for a frame, etc.
        uint32_t fpga_seq_number = get_fpga_seq_num(args->buf, bufferID);
        struct timeval frame_start_time = get_first_packet_recv_time(args->buf, bufferID);

        link_id = bufferID % args->num_links;

        for (int i = 0; i < args->num_data_sets; ++i) {
            // TODO Make this cleaner (single function)
            reorganize_32_to_16_feed_GPU_Correlated_Data( args->actual_num_freq,
                                                        args->actual_num_elements,
                                                          (int *)&args->buf->data[bufferID][i * (args->buf->buffer_size / args->num_data_sets)] );


            shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8(
                                                    args->actual_num_freq,
                                                    (int *)&args->buf->data[bufferID][i * (args->buf->buffer_size / args->num_data_sets)],
                                                    (complex_int_t *)&data_sets_buf[i * num_values * sizeof(complex_int_t)], link_id );
        }
        // TODO Add the ability to integrate the data down further here.

        // TODO Add flagging code here.
        for (int i = 0; i < num_values; ++i) {
            error_flags[i] = 0;
        }

        if (link_id + 1 == args->num_links) {

            for (int i = 0; i < args->num_data_sets; ++i) {
                // Send the frame.
                header->cpu_timestamp = frame_start_time;
                // TODO Is this the right thing to do?
                double time_offset = i * (args->num_timesamples * 2.56);
                header->cpu_timestamp.tv_usec = header->cpu_timestamp.tv_usec + time_offset;
                header->fpga_seq_number = fpga_seq_number*args->timesamples_per_packet + i * args->num_timesamples;
                header->num_freq = args->total_num_freq;
                header->num_vis = ((args->actual_num_elements * (args->actual_num_elements + 1)) / 2 );
                for (int j = 0; j < num_values; ++j) {
                    visibilities[j] = *(complex_int_t *)(data_sets_buf + i * (num_values * sizeof(complex_int_t)) + j * sizeof(complex_int_t));
                }

                //DEBUG("Sending frame to ch_master: FPGA_SEQ_NUMBER = %d ; NUM_FREQ = %d ; NUM_VIS = %d",
                //        header->fpga_seq_number, header->num_freq, header->num_vis);

                ssize_t bytes_sent = send(tcp_fd, buf, buffer_size, 0);
                if (bytes_sent == -1) {
                    ERROR("Could not send frame to ch_acq, error: %d", errno);
                    break;
                }
                if (bytes_sent != buffer_size) {
                    ERROR("Could not send all bytes: bytes sent = %d; buffer_size = %d", (int)bytes_sent, buffer_size);
                    break;
                }
            }
        }

        release_info_object(args->buf, bufferID);
        mark_buffer_empty(args->buf, bufferID);

        useableBufferIDs[0] = (useableBufferIDs[0] + 1) % args->buf->num_buffers;
    }

}
