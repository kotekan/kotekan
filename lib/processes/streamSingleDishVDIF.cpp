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
#include <sys/time.h>

#include "streamSingleDishVDIF.hpp"
#include "buffer.h"
#include "errors.h"

streamSingleDishVDIF::streamSingleDishVDIF(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&streamSingleDishVDIF::main_thread, this)){
    buf = get_buffer("vdif_in_buf");
    register_consumer(buf, unique_name.c_str());

    apply_config(0);
}
streamSingleDishVDIF::~streamSingleDishVDIF() {
}

void streamSingleDishVDIF::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    num_freq = config.get_int(unique_name,"num_freq");
    vdif_dest_port = config.get_int(unique_name,"vdif_dest_port");
    vdif_dest_ip = config.get_string(unique_name,"vdif_dest_ip");
}

void streamSingleDishVDIF::main_thread() {

    int frame_id = 0;
    uint8_t * frame = NULL;

    // Send files over the loop back address;
    // UDP variables
    struct sockaddr_in saddr_remote;
    int socket_fd;
    const size_t saddr_len = sizeof(saddr_remote);

    const int vdif_header_len = 32;

    // The *8 is to reduce the number of frames by sending a few per packet
    const int32_t packet_size = (vdif_header_len + num_freq);

    socket_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_fd == -1) {
        ERROR("Could not create socket for VDIF output stream");
        return;
    }

    memset((char *) &saddr_remote, 0, saddr_len);
    saddr_remote.sin_family = AF_INET;
    saddr_remote.sin_port = htons(vdif_dest_port);
    if (inet_aton(vdif_dest_ip.c_str(), &saddr_remote.sin_addr) == 0) {
        ERROR("Invalid address given for remote VDIF server: %s", vdif_dest_ip.c_str());
        exit(-1);
        return;
    }

    INFO ("Starting VDIF data stream thread to %s:%d", vdif_dest_ip.c_str(), vdif_dest_port);

    for(;;) {

        // Wait for a full buffer.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);

        // Check if the producer has finished, and we should exit.
        if (frame_id == -1) {
            break;
        }

        // Send data to remote server.
        // TODO rate limit this output
        for (int i = 0; i < buf->frame_size / packet_size; ++i) {

            int bytes_sent = sendto(socket_fd,
                             (void *)&frame[packet_size*i],
                             packet_size, 0,
                             (struct sockaddr *)&saddr_remote, saddr_len);

            if (bytes_sent == -1) {
                ERROR("Cannot set VDIF packet");
                break;
            }

            if (bytes_sent != packet_size) {
                ERROR("Did not send full vdif packet.");
            }

//            if (i%2048 == 0) usleep(200);
        }

        // Mark buffer as empty.
        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = ( frame_id + 1 ) % buf->num_frames;
    }
}
