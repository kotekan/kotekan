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

#include "networkInputPowerStream.hpp"
#include "util.h"
#include "errors.h"

networkInputPowerStream::networkInputPowerStream(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&networkInputPowerStream::main_thread, this))
    {

    buf = get_buffer("power_in_buf");

    //PER BUFFER
    string config_base = unique_name+string("stream_")+std::to_string(id);
    freqs = config.get_int(unique_name,"num_frequencies");
    elems = config.get_int(unique_name,"num_elements");

    port = config.get_int(unique_name,"port");
    server_ip = config.get_string(unique_name,"ip");
    protocol = config.get_string(unique_name,"protocol");

    atomic_flag_clear(&socket_lock);

    header.packet_length = freqs * sizeof(float);
    header.header_length = sizeof(IntensityPacketHeader);
    header.samples_per_packet = freqs;
    header.sample_type = 4;//uint32
    header.raw_cadence = 2.56e-6;
    header.num_freqs = freqs;
    header.num_elems = elems;
    header.samples_summed = 1;
    header.handshake_idx = -1;
    header.handshake_utc = -1;

    frame_idx=0;

}

networkInputPowerStream::~networkInputPowerStream() {
}

void networkInputPowerStream::apply_config(uint64_t fpga_seq) {
}

void networkInputPowerStream::receive_packet(void *buffer, int length){
    ssize_t rec = 0;
    while (rec < length) {
        int result = recv(socket_fd, ((char*)buffer) + rec, length - rec, 0);
        if (result == -1) {
            // Handle error ...
            break;
        }
        else if (result == 0) {
            // Handle disconnect ...
            break;
        }
        else {
            rec += result;
        }
    }
}

void networkInputPowerStream::main_thread() {
    int buf_id = 0;

    if (protocol == "UDP")
    {
    }
    else if (protocol == "TCP")
    {
        IntensityHeader handshake;

        struct sockaddr_in address; 
        memset(&address,0,sizeof(address));
        address.sin_addr.s_addr = htonl(INADDR_ANY);//inet_addr(server_ip.c_str());
        address.sin_port = htons(port);
        address.sin_family = AF_INET;

        if ((socket_fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
            ERROR("socket() failed");
        if (bind(socket_fd, (struct sockaddr *) &address, sizeof(address)) < 0)
            ERROR("bind() failed");
        if (listen(socket_fd, 1) < 0)
            ERROR("listen() failed");

        receive_packet(&handshake, sizeof(handshake));
        char *metadata = (char*)malloc(handshake.num_freqs * sizeof(float) * 2 + handshake.num_elems * sizeof(char));
        receive_packet(metadata, handshake.num_freqs * sizeof(float) * 2 + handshake.num_elems * sizeof(char));

        uint packet_length = handshake.num_freqs * sizeof(uint) + sizeof(IntensityPacketHeader);
        uint *recv_buffer = (uint*)malloc(packet_length);
        IntensityPacketHeader *pkt_header = (IntensityPacketHeader*)recv_buffer;
        uint *data = (uint*)(((char*)recv_buffer)+sizeof(IntensityPacketHeader));

        for (;;) {
            wait_for_empty_buffer(buf, buf_id);
            unsigned char* buf_ptr = buf->data[buf_id];

            receive_packet(recv_buffer, packet_length);
            for (int t = 0; t < times; t++) {
                for (int e = 0; e < elems; e++){
                    memcpy(buf_ptr,(void*)recv_buffer,packet_length);
                }
            }


            mark_buffer_full(buf, buf_id);
            buf_id = (buf_id + 1) % buf->num_buffers;
        }
        free(recv_buffer);
        free(metadata);
    }
    else ERROR("Bad protocol: %s\n", protocol.c_str());
}
