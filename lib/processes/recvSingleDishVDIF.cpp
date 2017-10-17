#include "recvSingleDishVDIF.hpp"
#include "errors.h"
#include "util.h"

inline bool file_exists(char * name) {
    struct stat buf;
    return (stat (name, &buf) == 0);
}

recvSingleDishVDIF::recvSingleDishVDIF(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&recvSingleDishVDIF::main_thread, this)) {

    buf = get_buffer("out_buf");
    register_producer(buf, unique_name.c_str());

    apply_config(0);
}

recvSingleDishVDIF::~recvSingleDishVDIF() {
}

void recvSingleDishVDIF::apply_config(uint64_t fpga_seq) {
    num_freq = config.get_int(unique_name,"num_freq");
    vdif_orig_port = config.get_int(unique_name,"vdif_orig_port");
    vdif_orig_ip = config.get_string(unique_name,"vdif_orig_ip");
}

void recvSingleDishVDIF::main_thread() {

    int file_num = 0;
    int frame_id = 0;
    uint8_t * frame = NULL;

    const int vdif_header_len = 32;

    // The *8 is to reduce the number of frames by sending a few per packet
    const int32_t packet_size = (vdif_header_len + num_freq);


    while(!stop_thread) {
        // Get an empty buffer to write into
        frame = wait_for_empty_frame(buf, unique_name.c_str(), frame_id);

        // Send data to remote server.
        // TODO rate limit this output
        for (int i = 0; i < buf->frame_size / packet_size; ++i) {
/*
            int bytes_recvd = sendto(socket_fd,
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
*/
        }

        INFO("recvSingleDishVDIF: marking buffer %s[%d] as full", buf->buffer_name, frame_id);
        mark_frame_full(buf, unique_name.c_str(), frame_id);

        file_num++;
        frame_id = (frame_id + 1) % buf->num_frames;
    }

    // Comment while support for finishing is not yet working.
    //mark_producer_done(&buf, 0);
}
