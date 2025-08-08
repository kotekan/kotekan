#include "vdifStream.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "util.h"              // for e_time

#include <arpa/inet.h>  // for inet_aton
#include <atomic>       // for atomic_bool
#include <errno.h>      // for errno
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <netinet/in.h> // for sockaddr_in, IPPROTO_UDP, htons
#include <regex>        // for match_results<>::_Base_type
#include <stdio.h>      // for size_t
#include <string.h>     // for memset, strerror
#include <string>       // for string, allocator
#include <sys/socket.h> // for sendto, socket, AF_INET, SOCK_DGRAM
#include <unistd.h>     // for usleep
#include <vector>       // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(vdifStream);

vdifStream::vdifStream(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&vdifStream::main_thread, this)) {

    buf = get_buffer("vdif_in_buf");
    buf->register_consumer(unique_name);
}
vdifStream::~vdifStream() {}

void vdifStream::main_thread() {

    // Apply config.
    _vdif_port = config.get<uint32_t>(unique_name, "vdif_port");
    _vdif_server_ip = config.get<std::string>(unique_name, "vdif_server_ip");

    int frame_id = {0};
    uint8_t* frame = nullptr;

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

    memset((char*)&saddr_remote, 0, saddr_len);
    saddr_remote.sin_family = AF_INET;
    saddr_remote.sin_port = htons(_vdif_port);
    if (inet_aton(_vdif_server_ip.c_str(), &saddr_remote.sin_addr) == 0) {
        ERROR("Invalid address given for remote VDIF server");
        return;
    }

    while (!stop_thread) {
        // IT - commented out to test performance without INFO calls.
        //        INFO("vdif_stream; waiting for full buffer to send, server_ip:{:s}:{:d}",
        //             _vdif_server_ip,
        //             _vdif_port);

        // Wait for a full buffer.
        frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;
        // IT - commented out to test performance without INFO calls.
        //        INFO("vdif_stream; got full buffer, sending to VDIF server.");

        start_t = e_time();

        // Send data to remote server.
        for (int i = 0; i < 16 * 625; ++i) {

            int bytes_sent = sendto(socket_fd, (void*)(&frame[packet_size * i]), packet_size, 0,
                                    (struct sockaddr*)&saddr_remote, saddr_len);

            if (i % 50 == 0) {
                usleep(sleep_period);
            }

            if (bytes_sent == -1) {
                ERROR("Cannot send VDIF packet, error: {:s}", strerror(errno));
                return;
            }

            if (bytes_sent != packet_size) {
                ERROR("Did not send full vdif packet.");
            }
        }

        diff_t = e_time() - start_t;
        INFO("vdif_stream: sent 1 seconds of vdif data to {:s}:{:d} in {:f} seconds; sleep set to "
             "{:d} microseconds",
             _vdif_server_ip, _vdif_port, diff_t, sleep_period);

        if (diff_t < 0.96) {
            sleep_period += 50;
        } else if (diff_t >= 0.99) {
            sleep_period -= 100;
        }

        // Mark buffer as empty.
        buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
    }
}
