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

networkPowerStream::networkPowerStream(Config& config, struct Buffer &buf_) :
    KotekanProcess(config, std::bind(&networkPowerStream::main_thread, this)),
    buf(buf_){

    //PER BUFFER
    times = config.get_int("/processing/samples_per_data_set") /
            config.get_int("/raw_capture/integration_length");
    freqs = config.get_int("/processing/num_local_freq");

    dest_port = config.get_int("/raw_capture/destination_port");
    dest_server_ip = config.get_string("/raw_capture/destination_ip");
    dest_protocol = config.get_string("/raw_capture/destination_protocol");

}

networkPowerStream::~networkPowerStream() {
}

void networkPowerStream::apply_config(uint64_t fpga_seq) {
}

void networkPowerStream::main_thread() {
    int buffer_id = 0;
    unsigned char local_data[freqs];


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
            buffer_id = get_full_buffer_from_list(&buf, &buffer_id, 1);

            for (int t=0; t<times; t++){
                for (int f=0; f<freqs; f++)
                    local_data[f] = ((float*)buf.data[buffer_id])[f]*4 +
                                        ((float*)buf.data[buffer_id])[f+freqs]*4;
                // Send data to remote server.
                int bytes_sent = sendto(socket_fd,
                                 (void *)local_data,
                                 freqs*sizeof(char), 0,
                                 (struct sockaddr *) &saddr_remote, sizeof(sockaddr_in));
                if (bytes_sent == -1) ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");
            }

            // Mark buffer as empty.
            mark_buffer_empty(&buf, buffer_id);
            buffer_id = (buffer_id + 1) % buf.num_buffers;
        }
    }
    else if (dest_protocol == "TCP")
    {
        // TCP variables
        for (;;) {
            // Wait for a full buffer.
            buffer_id = get_full_buffer_from_list(&buf, &buffer_id, 1);
            if (tcp_connected) {
                for (int t=0; t<times; t++){
                    for (int f=0; f<freqs; f++)
                        local_data[f] = ((float*)buf.data[buffer_id])[f]*4 +
                                            ((float*)buf.data[buffer_id])[f+freqs]*4;
                    // Send data to remote server.
                    int bytes_sent = send(socket_fd,
                                            (void *)local_data,
                                            freqs*sizeof(char),
                                            0);
                    if (bytes_sent == -1) {
                        close(socket_fd);
                        tcp_connected=false;
                    }
                }
            }
            else if (!tcp_connecting){
                tcp_connecting=true;
                connect_thread = std::thread(&networkPowerStream::tcpConnect, this);
                connect_thread.detach();
            }
            // Mark buffer as empty.
            mark_buffer_empty(&buf, buffer_id);
            buffer_id = (buffer_id + 1) % buf.num_buffers;
        }

    }
    else ERROR("Bad protocol: %s\n", dest_protocol.c_str());

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
    if (connect(socket_fd,(struct sockaddr *)&address,sizeof(address)) != 0){
//        ERROR("Error during TCP connection: %d\n",errno);
        tcp_connecting=false;
        close(socket_fd);
        return;
    }
    tcp_connected=true;
    tcp_connecting=false;


    struct timeval timeout;      
    timeout.tv_sec = 0;
    timeout.tv_usec = 200000;
    //TODO: handle errors
    setsockopt (socket_fd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout));
}



