#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <functional>

#include "vdifStream.hpp"
#include "util.h"
#include "errors.h"

vdifStream::vdifStream(Config& config, struct Buffer &buf_) :
    KotekanProcess(config, std::bind(&vdifStream::main_thread, this)),
    buf(buf_){
}
vdifStream::~vdifStream() {
}

void vdifStream::main_thread() {

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
        return;
    }

    memset((char *) &saddr_remote, 0, saddr_len);
    saddr_remote.sin_family = AF_INET;
    saddr_remote.sin_port = htons(config.beamforming.vdif_port);
    if (inet_aton(config.beamforming.vdif_server_ip, &saddr_remote.sin_addr) == 0) {
        ERROR("Invalid address given for remote VDIF server");
        return;
    }

    while(!stop_thread) {

        //INFO("vdif_stream; waiting for full buffer to send, server_ip:%s:%d",
        //     config.beamforming.vdif_server_ip,
        //     config.beamforming.vdif_port);

        // Wait for a full buffer.
        get_full_buffer_from_list(&buf, bufferID, 1);

        //INFO("vdif_stream; got full buffer, sending to VDIF server.");

        start_t = e_time();

        // Send data to remote server.
        for (int i = 0; i < 16*625; ++i) {

            int bytes_sent = sendto(socket_fd,
                             (void *)(buf.data[bufferID[0]][packet_size*i]),
                             packet_size, 0,
                             (struct sockaddr *) &saddr_remote, saddr_len);

            if (i % 50 == 0) {
                usleep(sleep_period);
            }

            if (bytes_sent == -1) {
                ERROR("Cannot set VDIF packet");
                return;
            }

            if (bytes_sent != packet_size) {
                ERROR("Did not send full vdif packet.");
            }
        }

        diff_t = e_time() - start_t;
        INFO("vdif_stream: sent 1 seconds of vdif data to %s:%d in %f seconds; sleep set to %d microseconds",
              config.beamforming.vdif_server_ip,
              config.beamforming.vdif_port,
              diff_t, sleep_period);

        if (diff_t < 0.96) {
            sleep_period += 50;
        } else if (diff_t >= 0.99) {
            sleep_period -= 100;
        }

        // Mark buffer as empty.
        mark_buffer_empty(&buf, bufferID[0]);
        bufferID[0] = (bufferID[0] + 1) % buf.num_buffers;
    }
}
