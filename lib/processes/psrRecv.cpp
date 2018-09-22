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

REGISTER_KOTEKAN_PROCESS(psrRecv);

psrRecv::psrRecv(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&psrRecv::main_thread, this))
    {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    port = config.get_int(unique_name,"port");
    server_ip = config.get_string(unique_name,"ip");



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
    int frame_id = 0;
    uint8_t * frame = NULL;

    int socket_fd;
    uint packet_length = freqs * sizeof(float) + sizeof(IntensityPacketHeader);

    struct sockaddr_in address;
    memset(&address,0,sizeof(address));
    address.sin_addr.s_addr = htonl(INADDR_ANY);//inet_addr(server_ip.c_str());
    address.sin_port = htons(port);
    address.sin_family = AF_INET;

    if ((socket_fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
        ERROR("socket() failed");
    if (bind(socket_fd, (struct sockaddr *) &address, sizeof(address)) < 0)
        ERROR("bind() failed");

    int max_packet_length = 65536;
    char *local_buf = (char*)calloc(max_packet_length,sizeof(char));

    int recv_depth = 3
    uint8_t *frame[recv_depth];
    uint *frame_id[recv_depth];

    while (!stop_thread) {
        frame[recv_depth-1] = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        uint32_t len = recvfrom(socket_fd,
                        local_buf,
                        packet_length, 0, NULL, 0);
        if (len != packet_length)
        {
            ERROR("BAD UDP PACKET! %i %i", len,errno)
        }
        else
        {
            extract desired frame = fpga_seq / frame_length

            if (desired_frame >= active_frame + recv_depth) {
                while (frame_id[0] < desired_frame) {
                    mark_frame_full(out_buf, unique_name.c_str(), frame_id[0]);
                    for (int i=1; i<recv_depth; i++) {
                        frame[i-1]    = frame[i];
                        frame_id[i-1] = frame_id[i];
                    }
                    frame_id[recv_depth-1] = (frame_id[recv_depth-1] + 1) % out_buf->num_frames;
                    if (!is_frame_empty(frame_id[rech_depth-1])) continue; //drop data until there's room!
                    frame[recv_depth-1] = wait_for_empty_frame(out_buf, unique_name.c_str(), frame_id);
                    initialize_frame(frame[recv_depth-1]);
                    
                }
            }

            put_data {
                memcpy(frame+t*elems*(freqs+1)*sizeof(uint)+
                       e*(freqs+1)*sizeof(uint),local_buf,packet_length);
            }

        }
    }

    else ERROR("Bad protocol: %s\n", protocol.c_str());
}
