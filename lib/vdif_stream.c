#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "vdif_stream.h"
#include "util.h"
#include "errors.h"

void exit_thread(int error) {
    pthread_exit((void*) &error);
}

void* vdif_stream(void * arg)
{
    struct VDIFstreamArgs * args = (struct VDIFstreamArgs *) arg;

    int bufferID[1] = {0};

    // UDP variables
    struct sockaddr_in saddr_remote;
    int socket_fd;
    const size_t saddr_len = sizeof(saddr_remote);

    const uint32_t packet_size = 5032;

    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd == -1) {
        ERROR("Could not create socket for VDIF output stream");
        exit_thread(-1);
    }

    memset((char *) &saddr_remote, 0, saddr_len);
    saddr_remote.sin_family = AF_INET;
    saddr_remote.sin_port = htons(args->config->beamforming.vdif_port);
    if (inet_aton(args->config->beamforming.vdif_server_ip, &saddr_remote.sin_addr) == 0) {
        ERROR("Invalid address given for remote VDIF server");
        exit_thread(-1);
    }

    for(EVER) {

        INFO("vdif_stream; waiting for full buffer to send, server_ip:%s:%d",
             args->config->beamforming.vdif_server_ip,
             args->config->beamforming.vdif_port);

        // Wait for a full buffer.
        get_full_buffer_from_list(args->buf, bufferID, 1);

        INFO("vdif_stream; got full buffer, sending to VDIF server.");

        // Send data to remote server.
        // TODO rate limit this output
        for (int i = 0; i < 16*625; ++i) {

            int bytes_sent = sendto(socket_fd,
                             (void *)&args->buf->data[bufferID[0]][packet_size*i],
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
        mark_buffer_empty(args->buf, bufferID[0]);
        bufferID[0] = (bufferID[0] + 1) % args->buf->num_buffers;
    }
}
