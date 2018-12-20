#include "networkInputPowerStream.hpp"

#include "errors.h"
#include "util.h"

#include <arpa/inet.h>
#include <functional>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

REGISTER_KOTEKAN_PROCESS(networkInputPowerStream);

networkInputPowerStream::networkInputPowerStream(Config& config, const string& unique_name,
                                                 bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&networkInputPowerStream::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

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
            ERROR("RECV = -1 %i", errno);
            // Handle error ...
            break;
        } else if (result == 0) {
            ERROR("RECV = 0 %i", errno);
            // Handle disconnect ...
            break;
        } else {
            rec += result;
        }
    }
}

void networkInputPowerStream::main_thread() {
    int frame_id = 0;
    uint8_t* frame = NULL;

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
            if (frame == NULL)
                break;

            for (int t = 0; t < times; t++) {
                for (int e = 0; e < elems; e++) {
                    uint32_t len = recvfrom(socket_fd, local_buf, packet_length, 0, NULL, 0);
                    if (len != packet_length) {
                        ERROR("BAD UDP PACKET! %i %i", len, errno)
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
            if (buf_ptr == NULL)
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
        ERROR("Bad protocol: %s\n", protocol.c_str());
}
