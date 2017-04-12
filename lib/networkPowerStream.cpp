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

}

networkPowerStream::~networkPowerStream() {
}

void networkPowerStream::apply_config(uint64_t fpga_seq) {
}

void networkPowerStream::main_thread() {
    int buffer_id = 0;
    unsigned char local_data[freqs];

    // UDP variables
    struct sockaddr_in saddr_remote;
    int socket_fd;
    const size_t saddr_len = sizeof(saddr_remote);

    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd == -1) {
        ERROR("Could not create socket for output stream");
        return;
    }

    memset((char *) &saddr_remote, 0, saddr_len);
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
                             (struct sockaddr *) &saddr_remote, saddr_len);

        }

        // Mark buffer as empty.
        mark_buffer_empty(&buf, buffer_id);
        buffer_id = (buffer_id + 1) % buf.num_buffers;
    }
}
