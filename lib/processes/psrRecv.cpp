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

#include "psrRecv.hpp"
#include "util.h"
#include "errors.h"
#include "vdif_functions.h"

REGISTER_KOTEKAN_PROCESS(psrRecv);

psrRecv::psrRecv(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&psrRecv::main_thread, this))
{
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    packet_length = config.get_int(unique_name,"udp_pulsar_packet_size");
    timesamples_per_packet = config.get_int(unique_name,"timesamples_per_packet");
    timesamples_per_frame = config.get_int(unique_name,"timesamples_per_frame");

    freqs_per_packet = config.get_int(unique_name,"freqs_per_packet");
    num_freq = config.get_int(unique_name,"num_freq");

    port = config.get_int(unique_name,"port");

//    out_buf = get_buffer("out_buf");
//    register_producer(out_buf, unique_name.c_str());
}

psrRecv::~psrRecv() {
}

void psrRecv::apply_config(uint64_t fpga_seq) {
}

void psrRecv::receive_packet(void *buffer, int length, int socket_fd){
    ssize_t rec = 0;
    while (rec < length) {
        int result = recv(socket_fd, ((char*)buffer) + rec, length - rec, 0);
        if (result == -1) {
            ERROR("RECV = -1 %i",errno);
            // Handle error ...
            break;
        }
        else if (result == 0) {
            ERROR("RECV = 0 %i",errno);
            // Handle disconnect ...
            break;
        }
        else {
            rec += result;
        }
    }
}

void psrRecv::main_thread() {
    int socket_fd;
    struct sockaddr_in address;
    memset(&address,0,sizeof(address));
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    address.sin_family = AF_INET;

    if ((socket_fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
        ERROR("socket() failed");
    if (bind(socket_fd, (struct sockaddr *) &address, sizeof(address)) < 0)
        ERROR("bind() failed");

    int max_packet_length = 65536;
    char *local_buf = (char*)calloc(max_packet_length,sizeof(char));

    int recv_depth = 30;
    assert(recv_depth < out_buf->num_frames);

    uint sample_idx0=0;
    int packets_per_frame = timesamples_per_frame / timesamples_per_packet;

    struct VDIFPacket {
        VDIFHeader h;
        uint8_t data[5000];
    };

    VDIFPacket *frame[recv_depth];
    uint frame_id[recv_depth];
    for (int i=0; i<recv_depth; i++) {
        frame_id[i] = i;
        frame[i] = (VDIFPacket*)wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id[i]);
        for (size_t t=0; t<packets_per_frame; t++)
            for (size_t f=0; f<num_freq/freqs_per_packet; f++)
                (frame[i] + (t*num_freq/freqs_per_packet + f))->h.invalid=true;
    }

    bool first_pass=true;

    while (!stop_thread) {
        uint32_t len = recvfrom(socket_fd,
                        local_buf,
                        max_packet_length, 0, NULL, 0);
        if (len != packet_length) {
            ERROR("BAD UDP PACKET! %i %i", len,errno);
            continue;
        }

        VDIFHeader *header = (VDIFHeader *)local_buf;
        DEBUG2("Header: \n"
             " seconds: %i\n"
             " legacy: %i\n"
             " invalid: %i\n"
             " data_frame: %i\n"
             " ref_epoch: %i\n"
             " unused: %i\n"
             " frame_len: %i\n"
             " log_num_chan: %i\n"
             " vdif_version: %i\n"
             " station_id: %c%c\n"
             " thread_id: %i\n"
             " bits_depth: %i\n"
             " data_type: %i\n"
             " eud1: %u\n"
             " edv: %i\n"
             " eud2: %u\n"
             " eud3: %u\n"
             " eud4: %u\n",
             header->seconds,
             header->legacy,
             header->invalid,
             header->data_frame,
             header->ref_epoch,
             header->unused,
             header->frame_len,
             header->log_num_chan,
             header->vdif_version,
             (char)(header->station_id>>8),(char)(header->station_id&0xff),
             header->thread_id,
             header->bits_depth,
             header->data_type,
             header->eud1,
             header->edv,
             header->eud2,
             header->eud3,
             header->eud4
             );
        if (first_pass) {
            sample_idx0 = header->eud3;
            first_pass=false;
        }

        if (header->eud3 < sample_idx0) continue; //drop the packet
        while (header->eud3 - sample_idx0 >= timesamples_per_frame*recv_depth) {
            int bad_packets = 0;
            for (int i=0; i<packets_per_frame; i++){
                for (size_t f=0; f<num_freq/freqs_per_packet; f++){
                    VDIFPacket *p = ((VDIFPacket*)frame[0]) + (i*num_freq/freqs_per_packet + f);
                    if (p->h.invalid) bad_packets++;
                }
            }
            INFO("Frame %02i closed, bad packets: %i (%2.1f\%)", frame_id[0], bad_packets, 
                    100*float(bad_packets)/packets_per_frame/(num_freq/freqs_per_packet));
            mark_frame_full(out_buf, unique_name.c_str(), frame_id[0]);
            for (int i=1; i < recv_depth; i++){
                frame[i-1] = frame[i];
                frame_id[i-1] = frame_id[i];
            }
            frame_id[recv_depth-1] = (frame_id[recv_depth-1]+1)%out_buf->num_frames;
            frame[recv_depth-1] = (VDIFPacket*)wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id[recv_depth-1]);
            for (int t=0; t<packets_per_frame; t++){
                for (size_t f=0; f<num_freq/freqs_per_packet; f++){
                    (frame[recv_depth-1] + (t*num_freq/freqs_per_packet + f))->h.invalid=true;
                }
            }
            sample_idx0 += timesamples_per_frame;
        }

        uint packet_idx = (header->eud3 - sample_idx0) / timesamples_per_packet % packets_per_frame;
        uint frame_idx  = (header->eud3 - sample_idx0) / timesamples_per_frame;
        DEBUG2("DEBUGGING: %i %i %i", header->eud3 - sample_idx0, packet_idx, frame_idx);
        VDIFPacket *dest = frame[frame_idx] + packet_idx*num_freq/freqs_per_packet + header->thread_id;

        memcpy(dest,local_buf,packet_length);
    }
}
