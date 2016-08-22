#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>

#include "stream_raw_vdif.h"
#include "buffers.h"
#include "errors.h"

void exit_thread(int error) {
    pthread_exit((void*) &error);
}

void *stream_raw_vdif(void * arg)
{
    struct stream_raw_vdif_arg * args = (struct stream_raw_vdif_arg *) arg;

    int useableBufferIDs[1] = {0};
    int bufferID = 0;

    // Check if the producer has finished, and we should exit.
    if (bufferID == -1) {
        exit_thread(0);
    }

    // UDP variables
    struct sockaddr_in saddr_remote;
    int socket_fd;
    const size_t saddr_len = sizeof(saddr_remote);

    // Max for jumbo frame.
    const uint32_t packet_size = 1056*8;

    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd == -1) {
        ERROR("Could not create socket for VDIF output stream");
        exit_thread(-1);
    }

    memset((char *) &saddr_remote, 0, saddr_len);
    saddr_remote.sin_family = AF_INET;
    saddr_remote.sin_port = htons(args->config->raw_cap.vdif_port);
    if (inet_aton(args->config->raw_cap.vdif_server, &saddr_remote.sin_addr) == 0) {
        ERROR("Invalid address given for remote VDIF server");
        exit_thread(-1);
    }

    for(;;) {

        // Wait for a full buffer.
        bufferID = get_full_buffer_from_list(args->buf, useableBufferIDs, 1);

        // Send data to remote server.
        // TODO rate limit this output
        for (int i = 0; i < args->buf->buffer_size / packet_size; ++i) {

            int bytes_sent = sendto(socket_fd,
                             (void *)&args->buf->data[bufferID][packet_size*i],
                             packet_size, 0,
                             &saddr_remote, saddr_len);

            if (bytes_sent == -1) {
                ERROR("Cannot set VDIF packet");
                exit_thread(-1);
            }

            if (bytes_sent != packet_size) {
                ERROR("Did not send full vdif packet.");
            }
        }

        // Mark buffer as empty.
        mark_buffer_empty(args->buf, bufferID);
        useableBufferIDs[0] = ( useableBufferIDs[0] + 1 ) % args->buf->num_buffers;
    }
}
