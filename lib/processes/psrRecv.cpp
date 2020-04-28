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
#include "chimeMetadata.h"

REGISTER_KOTEKAN_PROCESS(psrRecv);

/* A BUNCH OF CHIME-SPECIFIC PACKET PARAMETERS */
#define POLS_PER_PACKET 2
#define FREQS_PER_PACKET 4
#define TIMESAMPLES_PER_PACKET 625
#define SAMPLES_PER_SECOND 390625ull
/* A BUNCH OF CHIME-SPECIFIC PACKET PARAMETERS */

psrRecv::psrRecv(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&psrRecv::main_thread, this))
{
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    timesamples_per_frame = config.get<int>(unique_name,"timesamples_per_frame");
    num_vdif_threads = config.get<int>(unique_name,"num_vdif_threads");
    recv_depth = config.get<int>(unique_name,"recv_depth");
    port = config.get<int>(unique_name,"port");

    assert(recv_depth < out_buf->num_frames);
}

psrRecv::~psrRecv() {
}

void psrRecv::apply_config(uint64_t fpga_seq) {
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

    size_t packets_per_frame = timesamples_per_frame / TIMESAMPLES_PER_PACKET;

    struct VDIFPacket {
        VDIFHeader h;
        uint8_t data[POLS_PER_PACKET * FREQS_PER_PACKET * TIMESAMPLES_PER_PACKET];
    };
    VDIFHeader defaultHeader;

    VDIFPacket *frame[recv_depth];
    uint frame_id[recv_depth];
    for (int i=0; i<recv_depth; i++) {
        frame_id[i] = i;
        frame[i] = (VDIFPacket*)wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id[i]);
        if (frame[i] == NULL) return;
    }
    bool first_pass=true;

    uint64_t sample_idx0=0;
    while (!stop_thread) {
        uint32_t len = recvfrom(socket_fd,
                                local_buf,
                                max_packet_length, 0, NULL, 0);
        if (len != sizeof(VDIFPacket)) {
            ERROR("BAD UDP PACKET! %i %i", len,errno);
            continue;
        }

        VDIFHeader *header = (VDIFHeader *)local_buf;
        if (header->invalid) continue;
        uint64_t idx = header->seconds * SAMPLES_PER_SECOND + header->data_frame * TIMESAMPLES_PER_PACKET;
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
            sample_idx0 = idx;
            first_pass=false;
            defaultHeader.invalid      = 1;
            defaultHeader.ref_epoch    = header->ref_epoch;
            defaultHeader.frame_len    = header->frame_len;
            defaultHeader.log_num_chan = header->log_num_chan;
            defaultHeader.vdif_version = header->vdif_version;
            defaultHeader.station_id   = header->station_id;
            defaultHeader.bits_depth   = header->bits_depth;
            defaultHeader.data_type    = header->data_type;
            for (int i=0; i<recv_depth; i++) {
                uint64_t sample_idx = sample_idx0 + i*timesamples_per_frame;
                for (size_t t=0; t<packets_per_frame; t++)
                  for (size_t f=0; f<num_vdif_threads; f++){
                    VDIFHeader *hdr = &(frame[i] + (t*num_vdif_threads + f))->h;
                    memcpy(hdr,&defaultHeader,sizeof(VDIFHeader));
                    hdr->seconds    =  (t*TIMESAMPLES_PER_PACKET + sample_idx) / SAMPLES_PER_SECOND;
                    hdr->data_frame = ((t*TIMESAMPLES_PER_PACKET + sample_idx) % SAMPLES_PER_SECOND) / TIMESAMPLES_PER_PACKET;
                    hdr->thread_id  = f;
                  }
            }
        }

        if (idx < sample_idx0) continue; //drop the packet
        while (idx - sample_idx0 >= timesamples_per_frame*recv_depth) {
            int bad_packets = 0;
            for (size_t i=0; i<packets_per_frame; i++){
                for (size_t f=0; f<num_vdif_threads; f++){
                    VDIFPacket *p = ((VDIFPacket*)frame[0]) + (i*num_vdif_threads + f);
                    if (p->h.invalid) bad_packets++;
                }
            }
            DEBUG("Frame %02i closed, bad packets: %i (%2.1f\%)", frame_id[0], bad_packets, 
                    100*float(bad_packets)/packets_per_frame/num_vdif_threads);
            allocate_new_metadata_object(out_buf, frame_id[0]);
            atomic_add_lost_timesamples(out_buf, frame_id[0], bad_packets);
            mark_frame_full(out_buf, unique_name.c_str(), frame_id[0]);
            for (int i=1; i < recv_depth; i++){
                frame[i-1] = frame[i];
                frame_id[i-1] = frame_id[i];
            }
            frame_id[recv_depth-1] = (frame_id[recv_depth-1]+1)%out_buf->num_frames;
            frame[recv_depth-1] = (VDIFPacket*)wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id[recv_depth-1]);
            if (frame[recv_depth-1] == NULL) return;
            uint64_t sample_idx = sample_idx0 + recv_depth*timesamples_per_frame;
            for (size_t t=0; t<packets_per_frame; t++)
              for (size_t f=0; f<num_vdif_threads; f++){
                VDIFHeader *hdr = &(frame[recv_depth-1] + (t*num_vdif_threads + f))->h;
                memcpy(hdr,&defaultHeader,sizeof(VDIFHeader));
                hdr->seconds    =  (t*TIMESAMPLES_PER_PACKET + sample_idx) / SAMPLES_PER_SECOND;
                hdr->data_frame = ((t*TIMESAMPLES_PER_PACKET + sample_idx) % SAMPLES_PER_SECOND) / TIMESAMPLES_PER_PACKET;
                hdr->thread_id  = f;
              }
            sample_idx0 += timesamples_per_frame;
        }

        uint packet_idx = (idx - sample_idx0) / TIMESAMPLES_PER_PACKET % packets_per_frame;
        uint frame_idx  = (idx - sample_idx0) / timesamples_per_frame;
        header->edv = 0;
        header->eud1 = 0;
        header->eud2 = 0;
        header->eud3 = 0;
        header->eud4 = 0;
        VDIFPacket *dest = frame[frame_idx] + packet_idx*num_vdif_threads + header->thread_id;

        memcpy(dest,local_buf,sizeof(VDIFPacket));
    }
}
