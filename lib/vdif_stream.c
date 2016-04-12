#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
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

double e_time(void){
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

void* vdif_stream(void * arg)
{
    struct VDIFstreamArgs * args = (struct VDIFstreamArgs *) arg;

    int bufferID[1] = {0};

    double start_t, diff_t;
    int sleep_period = 3000;
    
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

        //INFO("vdif_stream; waiting for full buffer to send, server_ip:%s:%d",
        //     args->config->beamforming.vdif_server_ip,
        //     args->config->beamforming.vdif_port);

        // Wait for a full buffer.
        get_full_buffer_from_list(args->buf, bufferID, 1);

        //INFO("vdif_stream; got full buffer, sending to VDIF server.");

        start_t = e_time();

        // Send data to remote server.
        // TODO rate limit this output
        for (int i = 0; i < 16*625; ++i) {

            int bytes_sent = sendto(socket_fd,
                             (void *)&args->buf->data[bufferID[0]][packet_size*i],
                             packet_size, 0,
                             &saddr_remote, saddr_len);

            if (i % 50 == 0) {
                usleep(sleep_period);
            }

            if (bytes_sent == -1) {
                ERROR("Cannot set VDIF packet");
                exit_thread(-1);
            }

            if (bytes_sent != packet_size) {
                ERROR("Did not send full vdif packet.");
            }
        }

        diff_t = e_time() - start_t;
        INFO("vdif_stream: sent 1 seconds of vdif data to %s:%d in %f seconds; sleep set to %d microseconds",
              args->config->beamforming.vdif_server_ip,
              args->config->beamforming.vdif_port,
              diff_t, sleep_period);

        if (diff_t < 0.96) {
            sleep_period += 50;
        } else if (diff_t >= 0.99) {
            sleep_period -= 100;
        }

        // Mark buffer as empty.
        mark_buffer_empty(args->buf, bufferID[0]);
        bufferID[0] = (bufferID[0] + 1) % args->buf->num_buffers;
    }
}
