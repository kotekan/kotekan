#include "networkInputPowerStream.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for mark_frame_full, wait_for_empty_frame, register_producer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for ERROR
#include "powerStreamUtil.hpp" // for IntensityHeader, IntensityPacketHeader

#include <atomic>       // for atomic_bool
#include <errno.h>      // for errno
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <netinet/in.h> // for sockaddr_in, INADDR_ANY, htonl, htons, in_addr, IPPROTO_TCP
#include <regex>        // for match_results<>::_Base_type
#include <stdlib.h>     // for free, malloc, calloc
#include <string.h>     // for memcpy, memset
#include <string>       // for string, allocator, operator==
#include <sys/socket.h> // for bind, socket, accept, listen, recv, recvfrom, AF_INET
#include <sys/types.h>  // for uint, ssize_t
#include <vector>       // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(networkInputPowerStream);

networkInputPowerStream::networkInputPowerStream(Config& config, const std::string& unique_name,
                                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&networkInputPowerStream::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // PER BUFFER
    freqs = config.get<int>(unique_name, "num_freq");
    elems = config.get<int>(unique_name, "num_elements");

    port = config.get<uint32_t>(unique_name, "port");
    server_ip = config.get<std::string>(unique_name, "ip");
    protocol = config.get<std::string>(unique_name, "protocol");

    times = config.get<int>(unique_name, "samples_per_data_set")
            / config.get<int>(unique_name, "power_integration_length");
}

networkInputPowerStream::~networkInputPowerStream() {}

void networkInputPowerStream::receive_packet(void* buffer, int length, int socket_fd) {
    ssize_t rec = 0;
    while (rec < length) {
        int result = recv(socket_fd, ((char*)buffer) + rec, length - rec, 0);
        if (result == -1) {
            ERROR("RECV = -1 {:d}", errno);
            // Handle error ...
            break;
        } else if (result == 0) {
            ERROR("RECV = 0 {:d}", errno);
            // Handle disconnect ...
            break;
        } else {
            rec += result;
        }
    }
}

void networkInputPowerStream::main_thread() {
    int frame_id = 0;
    uint8_t* frame = nullptr;

    if (protocol == "UDP") {
        int socket_fd;
        uint packet_length = freqs * sizeof(float) + sizeof(IntensityPacketHeader);

        struct sockaddr_in address;
        memset(&address, 0, sizeof(address));
        address.sin_addr.s_addr = htonl(INADDR_ANY); // inet_addr(server_ip.c_str());
        address.sin_port = htons(port);
        address.sin_family = AF_INET;

        if ((socket_fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
            ERROR("socket() failed");
        if (bind(socket_fd, (struct sockaddr*)&address, sizeof(address)) < 0)
            ERROR("bind() failed");

        char* local_buf = (char*)calloc(packet_length, sizeof(char));

        while (!stop_thread) {
            frame = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id);
            if (frame == nullptr)
                break;

            for (int t = 0; t < times; t++) {
                for (int e = 0; e < elems; e++) {
                    uint32_t len =
                        recvfrom(socket_fd, local_buf, packet_length, 0, nullptr, nullptr);
                    if (len != packet_length) {
                        ERROR("BAD UDP PACKET! {:d} {:d}", len, errno)
                    } else {
                        memcpy(frame + t * elems * (freqs + 1) * sizeof(uint)
                                   + e * (freqs + 1) * sizeof(uint),
                               local_buf, packet_length);
                    }
                }
            }

            mark_frame_full(out_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % out_buf->num_frames;
        }

    } else if (protocol == "TCP") {
        IntensityHeader handshake;

        int socket_in;
        int socket_fd;
        int c;
        struct sockaddr_in address, client;
        memset(&address, 0, sizeof(address));
        address.sin_addr.s_addr = htonl(INADDR_ANY); // inet_addr(server_ip.c_str());
        address.sin_port = htons(port);
        address.sin_family = AF_INET;

        if ((socket_in = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
            ERROR("socket() failed");
        if (bind(socket_in, (struct sockaddr*)&address, sizeof(address)) < 0)
            ERROR("bind() failed");
        if (listen(socket_in, 1) < 0)
            ERROR("listen() failed");
        socket_fd = accept(socket_in, (struct sockaddr*)&client, (socklen_t*)&c);
        if (socket_fd < 0)
            ERROR("accept() failed");

        receive_packet(&handshake, sizeof(handshake), socket_fd);
        char* metadata = (char*)malloc(handshake.num_freqs * sizeof(float) * 2
                                       + handshake.num_elems * sizeof(char));
        receive_packet(metadata,
                       handshake.num_freqs * sizeof(float) * 2 + handshake.num_elems * sizeof(char),
                       socket_fd);

        uint packet_length = handshake.num_freqs * sizeof(uint) + sizeof(IntensityPacketHeader);
        uint* recv_buffer = (uint*)malloc(packet_length);
        IntensityPacketHeader* pkt_header = (IntensityPacketHeader*)recv_buffer;
        uint* data = (uint*)(((char*)recv_buffer) + sizeof(IntensityPacketHeader));

        while (!stop_thread) {
            unsigned char* buf_ptr =
                (unsigned char*)wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id);
            if (buf_ptr == nullptr)
                break;

            for (int t = 0; t < times; t++) {
                for (int e = 0; e < elems; e++) {
                    receive_packet(recv_buffer, packet_length, socket_fd);
                    //                    memcpy(buf_ptr,(void*)recv_buffer,packet_length);
                    memcpy(buf_ptr, (void*)data, freqs * sizeof(int));
                    ((int*)buf_ptr)[freqs] = pkt_header->samples_summed;
                    buf_ptr += (freqs + 1) * sizeof(int);
                }
            }
            mark_frame_full(out_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % out_buf->num_frames;
        }
        free(recv_buffer);
        free(metadata);
    } else
        ERROR("Bad protocol: {:s}\n", protocol);
}
