#include "recvSingleDishVDIF.hpp"
#include "errors.h"
#include "util.h"

recvSingleDishVDIF::recvSingleDishVDIF(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&recvSingleDishVDIF::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    apply_config(0);
}

recvSingleDishVDIF::~recvSingleDishVDIF() {
}

void recvSingleDishVDIF::apply_config(uint64_t fpga_seq) {
    num_freq = config.get_int(unique_name,"num_freq");
    orig_port = config.get_int(unique_name,"orig_port");
    orig_ip = config.get_string(unique_name,"orig_ip");
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
        frame = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        // Send data to remote server.
        // TODO rate limit this output
        for (int i = 0; i < out_buf->frame_size / packet_size; ++i) {
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

        INFO("recvSingleDishVDIF: marking buffer %s[%d] as full", out_buf->buffer_name, frame_id);
        mark_frame_full(out_buf, unique_name.c_str(), frame_id);

        file_num++;
        frame_id = (frame_id + 1) % out_buf->num_frames;
    }

    // Comment while support for finishing is not yet working.
    //mark_producer_done(&out_buf, 0);
}
