#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "rfiRecorder.hpp"
#include "buffer.h"
#include "errors.h"
#include "output_formating.h"
#include "chimeMetadata.h"

rfiRecorder::rfiRecorder(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rfiRecorder::main_thread, this)){
    rfi_buf = get_buffer("rfi_buf");
    register_consumer(rfi_buf, unique_name.c_str());    
}

rfiRecorder::~rfiRecorder() {
}

void rfiRecorder::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _sk_step = config.get_int(unique_name, "sk_step");
    _buf_depth = config.get_int(unique_name, "buffer_depth");
}
void rfiRecorder::main_thread() {
    apply_config(0);

    INFO("RFI RECORDER: Starting");
    int frame_ID = 0;
    bool First_Time = true;
    uint8_t slot_id, link_id;
    uint8_t * frame = nullptr;
    FILE *f;
    stream_id_t stream_ID;
    int64_t fpga_seq_num;
    int write_count = 0;
    // char write_buffer[10000];
    const int file_name_len = 100;
    char file_name[file_name_len];
    //write_buffer[0] = '\0';
    // Wait for, and drop full buffers
    while (!stop_thread) {

        frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_ID);
        // Check if the producer has finished, and we should exit.
        if (frame_ID == -1) {
            break;
        }
        fpga_seq_num = get_fpga_seq_num(rfi_buf, frame_ID);
        stream_ID = get_stream_id_t(rfi_buf, frame_ID);
        slot_id = stream_ID.slot_id;
        link_id = stream_ID.link_id;

        if(First_Time){
            snprintf(file_name, file_name_len, "../../scripts/rfi_recorder_%d_%d.rfi",slot_id,link_id);
            INFO("OPENING FILE: %s", file_name);
            f = fopen(file_name,"w");
            First_Time = false;
        }

        unsigned int * rfi_data = (unsigned int*)rfi_buf->frames[frame_ID];
        INFO("RFI RECORDER: Stream ID %d\n",stream_ID)

        for(int i = 0; i < _num_local_freq;i++){
            unsigned int counter = 0;
            for(int j = 0; j < _samples_per_data_set/_sk_step;j++){
                    counter += rfi_data[i + _num_local_freq*j];
            }
            int freq_bin = slot_id + link_id*16 + 128*i;
	        float rfi_perc = (float)counter/_samples_per_data_set;
	        int seq = (int)(fpga_seq_num & 0xFFFF);
            //float freq_mhz = 800 - freq_bin*((float)400/1024);
	        fwrite(&freq_bin,sizeof(int),1,f);
	        fwrite(&fpga_seq_num,sizeof(int64_t),1,f);
	        fwrite(&rfi_perc,sizeof(float),1,f);
	        INFO("Writing %d %d %f",freq_bin,seq,rfi_perc);
            
        }
        if(write_count==50){
            //fwrite(write_buffer,sizeof(char),strlen(write_buffer),f);
		    fclose(f);
            f = fopen(file_name,"a");
		    write_count = 0;
            //write_buffer[0] = '\0';
        }
        write_count++;

        mark_frame_empty(rfi_buf, unique_name.c_str(), frame_ID);
        frame_ID = (frame_ID + 1) % rfi_buf->num_frames;
    }
    fclose(f);
    INFO("RFI RECORDER: Closing Rfi Recorder");
}

