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

#include "networkPowerStream.hpp"
#include "util.h"
#include "errors.h"

networkPowerStream::networkPowerStream(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&networkPowerStream::main_thread, this)){

    buf = get_buffer("power_in_buf");

    //PER BUFFER
    times = config.get_int(unique_name, "samples_per_data_set") /
            config.get_int(unique_name, "integration_length");
    freqs = config.get_int(unique_name, "num_local_freq");
    elems = config.get_int(unique_name, "num_elements");

    dest_port = config.get_int(unique_name, "destination_port");
    dest_server_ip = config.get_string(unique_name, "destination_ip");
    dest_protocol = config.get_string(unique_name, "destination_protocol");

    atomic_flag_clear(&socket_lock);

    header.packet_length = freqs * sizeof(float);
    header.header_length = sizeof(IntensityPacketHeader);
    header.samples_per_packet = freqs;
    header.sample_type = 4;//uint32
    header.raw_cadence = 2.56e-6;
    header.num_freqs = freqs;
    header.num_elems = elems;
    header.samples_summed = config.get_int(unique_name, "integration_length");
    header.handshake_idx = -1;
    header.handshake_utc = -1;

    frame_idx=0;

}

networkPowerStream::~networkPowerStream() {
}

void networkPowerStream::apply_config(uint64_t fpga_seq) {
}

void networkPowerStream::main_thread() {
    int buffer_id = 0;
    uint packet_length = freqs * sizeof(float) + sizeof(IntensityPacketHeader);
    void *packet_buffer = malloc(packet_length);
        IntensityPacketHeader *packet_header = (IntensityPacketHeader *)packet_buffer;
        float *local_data = (float*)((char *)packet_buffer + sizeof(IntensityPacketHeader));
    struct timeval tv;

    if (dest_protocol == "UDP")
    {
        // UDP variables
        struct sockaddr_in saddr_remote;  /* the libc network address data structure */
        socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_fd == -1) {
            ERROR("Could not create UDP socket for output stream");
            return;
        }

        memset((char *) &saddr_remote, 0, sizeof(sockaddr_in));
        saddr_remote.sin_family = AF_INET;
        saddr_remote.sin_port = htons(dest_port);
        if (inet_aton(dest_server_ip.c_str(), &saddr_remote.sin_addr) == 0) {
            ERROR("Invalid address given for remote server");
            return;
        }
        INFO("%i %s",dest_port, dest_server_ip.c_str());

        for (;;) {
            // Wait for a full buffer.
            buffer_id = get_full_buffer_from_list(buf, &buffer_id, 1);

            for (int t=0; t<times; t++){
                for (int f=0; f<freqs; f++)
                    local_data[f] = ((float*)buf->data[buffer_id])[f]*4 +
                                        ((float*)buf->data[buffer_id])[f+freqs]*4;
                // Send data to remote server.
                int bytes_sent = sendto(socket_fd,
                                 (void*)local_data,
                                 freqs*sizeof(char), 0,
                                 (struct sockaddr *) &saddr_remote, sizeof(sockaddr_in));
                if (bytes_sent != freqs*sizeof(char))
                    ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");
            }

            // Mark buffer as empty.
            mark_buffer_empty(buf, buffer_id);
            buffer_id = (buffer_id + 1) % buf->num_buffers;
        }
    }
    else if (dest_protocol == "TCP")
    {
        // TCP variables
        for (;;) {
            // Wait for a full buffer.
            buffer_id = get_full_buffer_from_list(buf, &buffer_id, 1);
            while (atomic_flag_test_and_set(&socket_lock)) {}
            if (tcp_connected) {
                atomic_flag_clear(&socket_lock);
                for (int t=0; t<times; t++){
                    packet_header->frame_idx = frame_idx++;
                    for (int p=0; p<elems; p++){
                        packet_header->elem_idx = p;
                        packet_header->samples_summed = ((uint*)buf->data[buffer_id])[
                                                                t*elems*(freqs+1) + p*(freqs+1) + freqs];
                        memcpy(local_data,
                                buf->data[buffer_id]+t*elems*(freqs+1)*sizeof(uint)+
                                                          p*(freqs+1)*sizeof(uint),
                                freqs*sizeof(uint));
                        int bytes_sent = send(socket_fd,
                                                packet_buffer,
                                                packet_length,
                                                0);
                        if (bytes_sent != packet_length) {
                            while (atomic_flag_test_and_set(&socket_lock)) {}
                            close(socket_fd);
                            tcp_connected=false;
                            atomic_flag_clear(&socket_lock);
                            break;
                        }
                    }
                }
            }
            else if (!tcp_connecting){
                frame_idx+=times;
                handshake_idx = frame_idx;
                gettimeofday(&tv,NULL);
                handshake_utc =  tv.tv_sec + tv.tv_usec/1e6;
                tcp_connecting=true;
                atomic_flag_clear(&socket_lock);
                std::thread(&networkPowerStream::tcpConnect, this).detach();
            }
            else {
                frame_idx+=times;
                atomic_flag_clear(&socket_lock);
            }
            // Mark buffer as empty.
            mark_buffer_empty(buf, buffer_id);
            buffer_id = (buffer_id + 1) % buf->num_buffers;
        }

    }
    else ERROR("Bad protocol: %s\n", dest_protocol.c_str());

    free(packet_buffer);
}

void networkPowerStream::tcpConnect()
{
//    INFO("Connecting TCP Power Stream!");
    struct sockaddr_in address;
    address.sin_addr.s_addr = inet_addr(dest_server_ip.c_str());
    address.sin_port = htons(dest_port);
    address.sin_family = AF_INET;

    socket_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_fd == -1) {
        ERROR("Could not create TCP socket for output stream");
        return;
    }

    //TODO: handle errors, make dynamic
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 200000;

    if (connect(socket_fd,(struct sockaddr *)&address,sizeof(address)) != 0){
        while (atomic_flag_test_and_set(&socket_lock)) {}
        tcp_connecting=false;
        close(socket_fd);
        atomic_flag_clear(&socket_lock);
        return;
    }
    setsockopt (socket_fd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout));

    { //put together handshake
        while (atomic_flag_test_and_set(&socket_lock)) {}
            header.handshake_idx = handshake_idx;
            header.handshake_utc  = handshake_utc;
        atomic_flag_clear(&socket_lock);
        int bytes_sent = send(socket_fd,
                                (void *)&header,
                                sizeof(header),
                                0);
        if (bytes_sent != sizeof(header)) {
            ERROR("Could not send TCP header for output stream");
            while (atomic_flag_test_and_set(&socket_lock)) {}
            close(socket_fd);
            tcp_connected=false;
            atomic_flag_clear(&socket_lock);
            return;
        }
        //FIXME: remove hardcoding of freq & Stokes
        int info_size = freqs*2*sizeof(float) + elems*sizeof(char);
        void *info = malloc(info_size);
        for (int f=0; f<freqs; f++) {
            ((float*)info)[2*f]   = 800e6 - 400e6* f   /1024;
            ((float*)info)[2*f+1] = 800e6 - 400e6*(f+1)/1024;
        }
        // - description of stream (e.g. V / H pol, Stokes-I / Q / U / V)
        //  -8  -7  -6  -5  -4  -3  -2  -1  1   2   3   4
        //  YX  XY  YY  XX  LR  RL  LL  RR  I   Q   U   V
        for (int e=0; e<elems; e++)
            ((char*)((char*)info+info_size - elems*sizeof(char)))[e]=-5-e;
//            (((char*)info)+info_size - elems)[e]=-5-e;
        bytes_sent = send(socket_fd,
                                info,
                                info_size,
                                0);
        free(info);
        if (bytes_sent != info_size) {
            ERROR("Could not send TCP header for output stream");
            while (atomic_flag_test_and_set(&socket_lock)) {}
            close(socket_fd);
            tcp_connected=false;
            atomic_flag_clear(&socket_lock);
            return;
        }
    }
    while (atomic_flag_test_and_set(&socket_lock)) {}
    tcp_connected=true;
    tcp_connecting=false;
    atomic_flag_clear(&socket_lock);

}



