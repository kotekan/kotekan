#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>

#include "streamSingleDishVDIF.hpp"
#include "buffers.h"
#include "errors.h"

streamSingleDishVDIF::streamSingleDishVDIF(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&streamSingleDishVDIF::main_thread, this)){
    buf = get_buffer("vdif_in_buf");

    apply_config(0);
}
streamSingleDishVDIF::~streamSingleDishVDIF() {
}

void streamSingleDishVDIF::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    _num_freq = config.get_int("/raw_capture","num_freq");
    _vdif_port = config.get_int("/raw_capture","vdif_stream_port");
    _vdif_ip = config.get_string("/raw_capture","vdif_stream_ip");
}

void streamSingleDishVDIF::main_thread() {

    int bufferID = 0;

    // Send files over the loop back address;
    // UDP variables
    struct sockaddr_in saddr_remote;
    int socket_fd;
    const size_t saddr_len = sizeof(saddr_remote);

    const int vdif_header_len = 32;

    // The *8 is to reduce the number of frames by sending a few per packet
    const int32_t packet_size = (vdif_header_len + _num_freq) * 8;

    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd == -1) {
        ERROR("Could not create socket for VDIF output stream");
        return;
    }

    memset((char *) &saddr_remote, 0, saddr_len);
    saddr_remote.sin_family = AF_INET;
    saddr_remote.sin_port = htons(_vdif_port);
    if (inet_aton(_vdif_ip.c_str(), &saddr_remote.sin_addr) == 0) {
        ERROR("Invalid address given for remote VDIF server: %s", _vdif_ip.c_str());
        exit(-1);
        return;
    }

    INFO ("Starting VDIF data stream thread to %s:%d", _vdif_ip.c_str(), _vdif_port);

    for(;;) {

        // Wait for a full buffer.
        wait_for_empty_buffer(buf, unique_name.c_str(), bufferID);
        //bufferID = get_full_buffer_from_list(buf, &bufferID, 1);
        //INFO ("streamSingleDishVDIF: got buffer ID: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            break;
        }

        // Send data to remote server.
        // TODO rate limit this output
        for (int i = 0; i < buf->buffer_size / packet_size; ++i) {

            int bytes_sent = sendto(socket_fd,
                             (void *)buf->data[bufferID][packet_size*i],
                             packet_size, 0,
                             (struct sockaddr *)&saddr_remote, saddr_len);

            if (bytes_sent == -1) {
                ERROR("Cannot set VDIF packet");
                break;
            }

            if (bytes_sent != packet_size) {
                ERROR("Did not send full vdif packet.");
            }
        }

        // Mark buffer as empty.
        mark_buffer_empty(buf, unique_name.c_str(), bufferID);
        bufferID = ( bufferID + 1 ) % buf->num_buffers;
//        INFO ("Finished sending block of VDIF data to %s:%d", _vdif_ip.c_str(), _vdif_port);
    }
}
