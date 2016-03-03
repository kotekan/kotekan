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
#include "output_formating.h"
#include "config.h"

void* ch_acq_uplink_thread(void* arg)
{
    struct ch_acqUplinkThreadArg * args = (struct ch_acqUplinkThreadArg *) arg;

    struct Config * config = args->config;

    int buffer_ID = 0;

    // Connect to server.
    struct sockaddr_in ch_acq_addr;

    int tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd == -1) {
        ERROR("Could not create socket, errno: %d", errno);
        return NULL;
    }

    bzero(&ch_acq_addr, sizeof(ch_acq_addr));
    ch_acq_addr.sin_family = AF_INET;
    ch_acq_addr.sin_addr.s_addr = inet_addr(config->ch_master_network.collection_server_ip);
    ch_acq_addr.sin_port = htons(config->ch_master_network.collection_server_port);

    if (connect(tcp_fd, (struct sockaddr *)&ch_acq_addr, sizeof(ch_acq_addr)) == -1) {
        ERROR("Could not connect to collection server, error: %d", errno);
    }
    INFO("Connected to collection server on: %s:%d",
         config->ch_master_network.collection_server_ip,
         config->ch_master_network.collection_server_port);

    // Wait for, and transmit, full buffers.
    for (;;) {

        // This call is blocking!
        buffer_ID = get_full_buffer_from_list(args->buf, &buffer_ID, 1);

        // Check if the producer has finished, and we should exit.
        if (buffer_ID == -1) {
            INFO("Closing ch_acq_uplink");
            break;
        }

        // INFO("Sending TCP frame to ch_master. frame size: %d", args->buf->buffer_size);

        ssize_t bytes_sent = send(tcp_fd, args->buf->data[buffer_ID], args->buf->buffer_size, 0);
        if (bytes_sent <= 0) {
            ERROR("Could not send frame to ch_acq, error: %d", errno);
            break;
        }
        if (bytes_sent != args->buf->buffer_size) {
            ERROR("Could not send all bytes: bytes sent = %d; buffer_size = %d", (int)bytes_sent, args->buf->buffer_size);
            break;
        }
        INFO("Finished sending frame to ch_master");
        mark_buffer_empty(args->buf, buffer_ID);

        buffer_ID = (buffer_ID + 1) % args->buf->num_buffers;
    }
    int ret;
    pthread_exit((void *) &ret);
    return NULL;
}
