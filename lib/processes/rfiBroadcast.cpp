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

rfiBroadcast::rfiBroadcast(Config& config,
                                       const string& unique_name,
                                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rfiBroadcast::main_thread, this)){

    rfi_buf = get_buffer("rfi_buf");
    register_consumer(rfi_buf, unique_name.c_str());

    apply_config(0);
    srand(time(NULL));   // should only be called once

    atomic_flag_clear(&socket_lock);

    frame_idx=0;
}

rfiBroadcast::~rfiBroadcast() {
}

void rfiBroadcast::apply_config(uint64_t fpga_seq) {

    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _sk_step = config.get_int(unique_name, "sk_step");
    _buf_depth = config.get_int(unique_name, "buffer_depth");
    dest_port = config.get_int(unique_name, "destination_port");
    dest_server_ip = config.get_string(unique_name, "destination_ip");
    dest_protocol = config.get_string(unique_name, "destination_protocol");
    frames_per_packet = config.get_int(unique_name, "frames_per_packet");
}

void rfiBroadcast::main_thread() {

    int frame_id = 0;
    int64_t dummy_seq = 0;
    int packet_length = frames_per_packet * _num_local_freq*(sizeof(int) + sizeof(int64_t) + sizeof(float));
    char *packet_buffer = (char *)malloc(packet_length);
    int data_counter = 0;
    int bytes_written = 0;
    uint8_t * frame = nullptr;
    if (dest_protocol == "UDP")
    {
        // UDP variables
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

        INFO("RFI BROADCAST: UDP Connection: %i %s",dest_port, dest_server_ip.c_str());

        for (;;) {
            INFO("RFI BROADCAST: Waiting for Frame!")
            frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
            INFO("RFI BROADCAST: Got Frame!")
            data_counter++;

            //READ BUFFER
            unsigned int * rfi_data = (unsigned int*)rfi_buf->frames[frame_id];
            fpga_seq_num = get_fpga_seq_num(rfi_buf, frame_id);
            stream_ID = get_stream_id_t(rfi_buf, frame_id);
            slot_id = stream_ID.slot_id;
            link_id = stream_ID.link_id;
            

            //INFO("RFI BROADCAST: Reading RFI data... %d",slot_id);
            for(int i = 0; i < _num_local_freq; i++){

                unsigned int counter = 0;
                for(int j = 0; j < _samples_per_data_set/_sk_step; j++){
                        counter += rfi_data[i + _num_local_freq*j];
                }
                int freq_bin = slot_id + link_id*16 + 128*i;
                float rfi_perc = (float)counter/_samples_per_data_set;
                //rfi_perc = (float)rand()/RAND_MAX; // Dummy Line
                
                //Add info to packet
                memcpy(packet_buffer + bytes_written, &freq_bin, sizeof(int));
                bytes_written += sizeof(int);
                memcpy(packet_buffer + bytes_written, &fpga_seq_num, sizeof(int64_t));
                bytes_written += sizeof(int64_t);
                memcpy(packet_buffer + bytes_written, &rfi_perc, sizeof(float));
                bytes_written += sizeof(float);

            }
            if(data_counter == frames_per_packet){
                //Send Packet
                int bytes_sent = sendto(socket_fd,
                                 packet_buffer,
                                 bytes_written, 0,
                                 (struct sockaddr *) &saddr_remote, sizeof(sockaddr_in));

                if (bytes_sent != bytes_written){
                    ERROR("SOMETHING WENT WRONG IN UDP TRANSMIT");
                }

                INFO("RFI BROADCAST: Sending %d Bytes... Freq Bin %d",bytes_written);
                bytes_written = 0;
                data_counter= 0;
            }

            //dummy_seq += 32768;// Dummy Line

            // Mark buffer as empty.
            mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % rfi_buf->num_frames;

        }
    }
    //TODO ADD real TCP support, currently doesn't work
    /*
    else if (dest_protocol == "TCP")
    {
        for (;;) 
        {
            INFO("RFI BROADCAST: Getting RFI Buffer, TCP Status %d",tcp_connected);
            // Wait for a full buffer.
            buffer_id = wait_for_full_buffer(rfi_buf, unique_name.c_str(), buffer_id);

            while (atomic_flag_test_and_set(&socket_lock)) {}

            if (tcp_connected) {
                data_counter++;
                atomic_flag_clear(&socket_lock);
                //READ BUFFER
                stream_ID = rfi_buf->info[buffer_id]->stream_ID;
                fpga_seq_num = rfi_buf->info[buffer_id]->fpga_seq_num;
                slot_id = (stream_ID & 0x00F0) >> 4;
                link_id = stream_ID & 0x000F;
                unsigned int * rfi_data = (unsigned int*)rfi_buf->data[buffer_id];
                INFO("RFI BROADCAST: Reading RFI data...");
                for(int i = 0; i < _num_local_freq; i++){
                    unsigned int counter = 0;
                    for(int j = 0; j < _samples_per_data_set/_sk_step; j++){
                            counter += rfi_data[i + _num_local_freq*j];
                    }
                    int freq_bin = slot_id + link_id*16 + 128*i;
                    float rfi_perc = (float)counter/_samples_per_data_set;
                    fpga_seq_num = dummy_seq;
                    rfi_perc = (float)rand()/RAND_MAX;
                    
                    memcpy(packet_buffer + bytes_written, &freq_bin, sizeof(int));
                    bytes_written += sizeof(int);
                    memcpy(packet_buffer + bytes_written, &fpga_seq_num, sizeof(int64_t));
                    bytes_written += sizeof(int64_t);
                    memcpy(packet_buffer + bytes_written, &rfi_perc, sizeof(float));
                    bytes_written += sizeof(float);
                    INFO("BYTES WRITTEN: %d",bytes_written);
                }
                //SEND DATA
                if(data_counter == frames_per_packet){
                    data_counter = 0;
                    bytes_written = 0;
                    INFO("RFI BROADCAST: Sending %d Bytes... ",packet_length);
                    int bytes_sent = send(socket_fd,
                                            packet_buffer,
                                            packet_length,
                                            0);
                    if (bytes_sent != packet_length) {
                        while (atomic_flag_test_and_set(&socket_lock)) {}
                        close(socket_fd);
                        tcp_connected=false;
                        atomic_flag_clear(&socket_lock);
                        break;
                    }
                }
                dummy_seq += 32768;
                
            }
            else if (!tcp_connecting){;
                tcp_connecting=true;
                atomic_flag_clear(&socket_lock);
                INFO("RFI BROADCAST: reconnecting...");
                std::thread(&rfiBroadcast::tcpConnect, this).detach();
            }
            else {
                INFO("RFI BROADCAST: Still trying to Connect...");
                atomic_flag_clear(&socket_lock);
            }
            // Mark buffer as empty.
            release_info_object(rfi_buf, buffer_id);
            mark_buffer_empty(rfi_buf, unique_name.c_str(), buffer_id);
            buffer_id = (buffer_id + 1) % rfi_buf->num_buffers;
        }

    }*/
    
    else{
        ERROR("Bad protocol: %s\n", dest_protocol.c_str());
    }

    free(packet_buffer);
}

void rfiBroadcast::tcpConnect()
{
    
    struct sockaddr_in address;
    address.sin_addr.s_addr = inet_addr(dest_server_ip.c_str());
    address.sin_port = htons(dest_port);
    address.sin_family = AF_INET;
    INFO("RFI BROADCAST: Connecting TCP Power Stream! Addr: %s Port %d",dest_server_ip.c_str(),dest_port);
    socket_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (socket_fd == -1) {
        ERROR("Could not create TCP socket for output stream");
        return;
    }
    int err = connect(socket_fd,(struct sockaddr *)&address,sizeof(address));
    if (err != 0){
        INFO("RFI BROADCAST: Could Not Connect Error Code: %s",strerror(errno));
        while (atomic_flag_test_and_set(&socket_lock)) {}
        tcp_connecting=false;
        close(socket_fd);
        atomic_flag_clear(&socket_lock);
        return;
    }
    
    { //put together handshake
        
        while (atomic_flag_test_and_set(&socket_lock)) {}
        atomic_flag_clear(&socket_lock);
    }
    while (atomic_flag_test_and_set(&socket_lock)) {}
    INFO("RFI BROADCAST: Connected");
    tcp_connected=true;
    tcp_connecting=false;
    atomic_flag_clear(&socket_lock);
}



