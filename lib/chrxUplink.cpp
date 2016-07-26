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

#include "chrxUplink.hpp"
#include "buffers.h"
#include "errors.h"
#include "output_formating.h"
#include "config.h"

chrxUplink::chrxUplink(struct Config &config_,
                  struct Buffer &vis_buf_,
                  struct Buffer &gate_buf_) :
                  KotekanProcess(config_, std::bind(&chrxUplink::main_thread, this)),
                  vis_buf(vis_buf_),
                  gate_buf(gate_buf_) {
}

chrxUplink::~chrxUplink() {
}

void chrxUplink::main_thread() {
    int buffer_ID = 0;

    // Connect to server.
    struct sockaddr_in ch_acq_addr;

    int tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd == -1) {
        ERROR("Could not create socket, errno: %d", errno);
        return;
    }

    bzero(&ch_acq_addr, sizeof(ch_acq_addr));
    ch_acq_addr.sin_family = AF_INET;
    ch_acq_addr.sin_addr.s_addr = inet_addr(config.ch_master_network.collection_server_ip);
    ch_acq_addr.sin_port = htons(config.ch_master_network.collection_server_port);

    if (connect(tcp_fd, (struct sockaddr *)&ch_acq_addr, sizeof(ch_acq_addr)) == -1) {
        ERROR("Could not connect to collection server, error: %d", errno);
    }
    INFO("Connected to collection server on: %s:%d",
         config.ch_master_network.collection_server_ip,
         config.ch_master_network.collection_server_port);

    // Wait for, and transmit, full buffers.
    while(!stop_thread) {

        // This call is blocking!
        buffer_ID = get_full_buffer_from_list(&vis_buf, &buffer_ID, 1);

        // Check if the producer has finished, and we should exit.
        if (buffer_ID == -1) {
            INFO("Closing ch_acq_uplink");
            break;
        }

        // INFO("Sending TCP frame to ch_master. frame size: %d", vis_buf->buffer_size);

        ssize_t bytes_sent = send(tcp_fd,vis_buf.data[buffer_ID], vis_buf.buffer_size, 0);
        if (bytes_sent <= 0) {
            ERROR("Could not send frame to ch_acq, error: %d", errno);
            break;
        }
        if (bytes_sent != vis_buf.buffer_size) {
            ERROR("Could not send all bytes: bytes sent = %d; buffer_size = %d",
                    (int)bytes_sent, vis_buf.buffer_size);
            break;
        }
        INFO("Finished sending frame to ch_master");

        if (config.gating.enable_basic_gating == 1) {
            DEBUG("Getting gated buffer");
            get_full_buffer_from_list(&gate_buf, &buffer_ID, 1);

            DEBUG("Sending gated buffer");
            bytes_sent = send(tcp_fd, gate_buf.data[buffer_ID], gate_buf.buffer_size, 0);
            if (bytes_sent <= 0) {
                ERROR("Could not send gated date frame to ch_acq, error: %d", errno);
                break;
            }
            if (bytes_sent != gate_buf.buffer_size) {
                ERROR("Could not send all bytes in gated data frame: bytes sent = %d; buffer_size = %d",
                        (int)bytes_sent, gate_buf.buffer_size);
                break;
            }
            INFO("Finished sending gated data frame to ch_master");
            mark_buffer_empty(&gate_buf, buffer_ID);
        }

        mark_buffer_empty(&vis_buf, buffer_ID);

        buffer_ID = (buffer_ID + 1) % vis_buf.num_buffers;
    }
}
