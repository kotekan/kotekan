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
#include <errno.h>
#include <time.h>
#include "rfiBroadcast.hpp"
#include "util.h"
#include "errors.h"
#include "chimeMetadata.h"

REGISTER_KOTEKAN_PROCESS(rfiBroadcast);

rfiBroadcast::rfiBroadcast(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rfiBroadcast::main_thread, this)){
    //Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    //Register process as consumer
    register_consumer(rfi_buf, unique_name.c_str());
    //Intialize internal config
    apply_config(0);
}

rfiBroadcast::~rfiBroadcast() {
}

void rfiBroadcast::apply_config(uint64_t fpga_seq) {

    _num_freq = config.get_int(unique_name, "num_total_freq");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    _sk_step = config.get_int(unique_name, "sk_step");
    COMBINED = config.get_bool(unique_name,"rfi_combined");
    frames_per_packet = config.get_int(unique_name, "frames_per_packet");

    total_links = config.get_int(unique_name, "total_links");
    dest_port = config.get_int(unique_name, "destination_port");
    dest_server_ip = config.get_string(unique_name, "destination_ip");
    dest_protocol = config.get_string(unique_name, "destination_protocol");
}

void rfiBroadcast::main_thread() {

    uint32_t frame_id = 0;
    uint32_t i, j, f;
    uint32_t bytes_sent = 0;
    uint8_t * frame = NULL;
    uint32_t link_id = 0;
    uint16_t StreamIDs[total_links];
    //Intialize empty packet header
    struct RFIHeader rfi_header = {.rfi_combined=(uint8_t)COMBINED, .sk_step=_sk_step, .num_elements=_num_elements, .samples_per_data_set=_samples_per_data_set,
                      .num_total_freq=1024, .num_local_freq=_num_local_freq, frames_per_packet=frames_per_packet};
    //Intialize empty packet
    uint32_t packet_length = sizeof(rfi_header) + _num_local_freq*sizeof(float);
    char *packet_buffer = (char *)malloc(packet_length);

    if (dest_protocol == "UDP")
    {
        // UDP Stuff
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
        //Connection succesful
        INFO("UDP Connection: %i %s",dest_port, dest_server_ip.c_str());

        while (!stop_thread) { //Endless loop

            float rfi_data [total_links][_num_local_freq*_samples_per_data_set/_sk_step];
            float RFI_Avg[total_links][_num_local_freq];
            //Zero Average array
            for(i = 0; i < total_links; i++){
                for(j = 0; j < _num_local_freq; j++){
                    RFI_Avg[i][j] = 0;
                }
            }
            //Loop through all frames that should be averages together
            for(f = 0; f < frames_per_packet*total_links; f++){
                //Get Frame
                frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
                if (frame == NULL) break;
                //Copy frame data to array
                memcpy(rfi_data[link_id], frame, rfi_buf->frame_size);
                if(f == 0){
                    //Adjust Header on initial frame
                    rfi_header.seq_num = get_fpga_seq_num(rfi_buf, frame_id);
                }
                //Adjust Stream ID's
                StreamIDs[link_id] = get_stream_id(rfi_buf, frame_id);
                //Sum over the whole frame
                for(i = 0; i < _num_local_freq; i++){
                    for(j = 0; j < _samples_per_data_set/_sk_step; j++){
                        RFI_Avg[link_id][i] += rfi_data[link_id][i + _num_local_freq*j];
                    }
                }
                //Mark Frame Empty
                mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
                frame_id = (frame_id + 1) % rfi_buf->num_frames;
                link_id = (link_id + 1) % total_links;
            }
            double start_time = e_time();
            //Loop through each link to send data seperately
            for(j = 0; j < total_links; j++){
                //Normalize Sum (Take Average)
                for(i = 0; i < _num_local_freq; i++){
                    RFI_Avg[j][i] /= frames_per_packet*(_samples_per_data_set/_sk_step);
                    if(i == 0){ DEBUG("SK value %f for freq %d, stream %d", RFI_Avg[j][i], i, StreamIDs[j])}
                }
                //Add Stream ID to header
                rfi_header.streamID = StreamIDs[j];
                //Add Header to packet
                memcpy(packet_buffer, &rfi_header, sizeof(rfi_header));
                //Add Data to packet
                memcpy(packet_buffer + sizeof(rfi_header), RFI_Avg[j], _num_local_freq*sizeof(float));
                //Send Packet
                bytes_sent = sendto(socket_fd,
                                 packet_buffer,
                                 packet_length, 0,
                                 (struct sockaddr *) &saddr_remote, sizeof(sockaddr_in));
                //Check if packet sent properly
                if (bytes_sent != packet_length){ ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");}
            }
            INFO("Frame ID %d Succesfully Broadcasted %d links of %d Bytes in %fms",frame_id, total_links, bytes_sent, (e_time()-start_time)*1000);
        }
    }
    else{
        ERROR("Bad protocol: %s Only UDP currently Supported", dest_protocol.c_str());
    }
    free(packet_buffer);
}

