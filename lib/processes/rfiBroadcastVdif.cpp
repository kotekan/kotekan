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
#include "rfiBroadcastVdif.hpp"
#include "util.h"
#include "errors.h"

rfiBroadcastVdif::rfiBroadcastVdif(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rfiBroadcastVdif::main_thread, this)){
    //Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    //Register process as consumer
    register_consumer(rfi_buf, unique_name.c_str());
    //Intialize internal config
    apply_config(0);
}

rfiBroadcastVdif::~rfiBroadcastVdif() {
}

void rfiBroadcastVdif::apply_config(uint64_t fpga_seq) {

    _num_freq = config.get_int(unique_name, "num_freq");
    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _sk_step = config.get_int(unique_name, "sk_step");
    COMBINED = config.get_bool(unique_name,"RFI_combined");

    dest_port = config.get_int(unique_name, "destination_port");
    dest_server_ip = config.get_string(unique_name, "destination_ip");
    dest_protocol = config.get_string(unique_name, "destination_protocol");
}

void rfiBroadcastVdif::main_thread() {

    int frame_id = 0;
    unsigned int i, j;
    uint8_t * frame = NULL;

    //Intialize empty packet header
    int packet_header_length = sizeof(bool) + sizeof(int)*4 + sizeof(uint32_t);
    char *packet_header = (char *)malloc(packet_header_length);
    
    //Declare array to hold averaged kurtosis estimates
    float RFI_Avg[_num_freq];
    unsigned int packet_header_bytes_written = 0;
    uint32_t vdif_seq_num = 0;

    //Intialize empty packet
    int packet_length = packet_header_length + _num_freq*sizeof(float);
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

        //Fill Header
        memcpy(packet_header + packet_header_bytes_written, &COMBINED, sizeof(bool));
        packet_header_bytes_written += sizeof(bool);
        memcpy(packet_header + packet_header_bytes_written, &_sk_step, sizeof(int));
        packet_header_bytes_written += sizeof(int);
        memcpy(packet_header + packet_header_bytes_written, &_num_elements, sizeof(int));
        packet_header_bytes_written += sizeof(int);
        memcpy(packet_header + packet_header_bytes_written, &_samples_per_data_set, sizeof(int));
        packet_header_bytes_written += sizeof(int);
        memcpy(packet_header + packet_header_bytes_written, &_num_freq, sizeof(int));
        packet_header_bytes_written += sizeof(int);
        
        //Connection succesful
        INFO("UDP Connection: %i %s",dest_port, dest_server_ip.c_str());
        
        while (!stop_thread) { //Endless loop

            //Get Frame
            frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
            if (frame == NULL) break;

            float rfi_data[_num_freq*_samples_per_data_set/_sk_step];
            memcpy(rfi_data, frame, rfi_buf->frame_size);

            //Adjust Header
            memcpy(packet_header + packet_header_bytes_written, &vdif_seq_num, sizeof(uint32_t));
            packet_header_bytes_written += sizeof(uint32_t);
            
            //Add Header to packet
            memcpy(packet_buffer, packet_header, packet_header_length);

            for(i = 0; i < _num_freq; i++){
                RFI_Avg[i] = 0;
                //Average over the whole frame
                for(j = 0; j < _samples_per_data_set/_sk_step; j++){
                    RFI_Avg[i] += rfi_data[i + _num_freq*j];
                    //INFO("SK value %f for freq %d", rfi_data[i + _num_freq*j], i)
                }
                RFI_Avg[i] /= (_samples_per_data_set/_sk_step);
            }

            //Add Data to packet
            memcpy(packet_buffer + packet_header_length, RFI_Avg, _num_freq*sizeof(float));
                
            //Send Packet
            int bytes_sent = sendto(socket_fd,
                             packet_buffer,
                             packet_length, 0,
                             (struct sockaddr *) &saddr_remote, sizeof(sockaddr_in));

            if (bytes_sent != packet_length){
                ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");
            }
            
            //Prepare Header For Adjustment
            packet_header_bytes_written -= sizeof(uint32_t); 
            
            //Adjust Packet Number
            vdif_seq_num++;

            // Mark frame as empty.
            mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % rfi_buf->num_frames;
            
            INFO("Frame ID %d Succesfully Broadcasted %d",frame_id, bytes_sent);
        }
    }
    else{

        ERROR("Bad protocol: %s Only UDP currently Supported", dest_protocol.c_str());

    }

    free(packet_buffer);
}




