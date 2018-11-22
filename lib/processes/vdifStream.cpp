#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <functional>
#include <string>

#include "vdifStream.hpp"
#include "util.h"
#include "errors.h"

REGISTER_KOTEKAN_PROCESS(vdifStream);

vdifStream::vdifStream(Config& config, const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&vdifStream::main_thread, this)) {

    buf = get_buffer("vdif_in_buf");
    register_consumer(buf, unique_name.c_str());
}
vdifStream::~vdifStream() {
}

void vdifStream::main_thread() {

    // Apply config.
    _vdif_port = config.get<uint32_t>(unique_name, "vdif_port");
    _vdif_server_ip = config.get<std::string>(unique_name, "vdif_server_ip");

    int frame_id = {0};
    uint8_t * frame = NULL;

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
    saddr_remote.sin_port = htons(_vdif_port);
    if (inet_aton(_vdif_server_ip.c_str(), &saddr_remote.sin_addr) == 0) {
        ERROR("Invalid address given for remote VDIF server");
        return;
    }

    while(!stop_thread) {
//IT - commented out to test performance without INFO calls.
//        INFO("vdif_stream; waiting for full buffer to send, server_ip:%s:%d",
//             _vdif_server_ip.c_str(),
//             _vdif_port);

        // Wait for a full buffer.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;
//IT - commented out to test performance without INFO calls.
//        INFO("vdif_stream; got full buffer, sending to VDIF server.");

        start_t = e_time();

        // Send data to remote server.
        for (int i = 0; i < 16*625; ++i) {

            int bytes_sent = sendto(socket_fd,
                             (void *)(&frame[packet_size*i]),
                             packet_size, 0,
                             (struct sockaddr *) &saddr_remote, saddr_len);

            if (i % 50 == 0) {
                usleep(sleep_period);
            }

            if (bytes_sent == -1) {
                ERROR("Cannot send VDIF packet, error: %s", strerror(errno));
                return;
            }

            if (bytes_sent != packet_size) {
                ERROR("Did not send full vdif packet.");
            }
        }

        diff_t = e_time() - start_t;
        INFO("vdif_stream: sent 1 seconds of vdif data to %s:%d in %f seconds; sleep set to %d microseconds",
              _vdif_server_ip.c_str(),
              _vdif_port,
              diff_t, sleep_period);

        if (diff_t < 0.96) {
            sleep_period += 50;
        } else if (diff_t >= 0.99) {
            sleep_period -= 100;
        }

        // Mark buffer as empty.
        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }
}
