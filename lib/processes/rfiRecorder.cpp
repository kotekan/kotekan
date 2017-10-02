#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>

#include "rfiRecorder.hpp"
#include "buffers.h"
#include "errors.h"
#include "output_formating.h"

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
    int buffer_ID = 0;
    
    // Wait for, and drop full buffers
    while (!stop_thread) {

        buffer_ID = wait_for_full_buffer(rfi_buf, unique_name.c_str(), buffer_ID);
        // Check if the producer has finished, and we should exit.
        if (buffer_ID == -1) {
            break;
        }
	unsigned int * rfi_data = (unsigned int*)rfi_buf->data[buffer_ID];
	uint16_t stream_ID = rfi_buf->info[buffer_ID]->stream_ID;
	int slot_id = (stream_ID & 0x00F0) >> 4;
	int link_id = stream_ID & 0x000F;
	const int file_name_len = 100;
        char file_name[file_name_len];
        snprintf(file_name, file_name_len, "../../scripts/rfi_recorder_%d.rfi",slot_id);
	INFO("RFI RECORDER: Stream ID %d\n",stream_ID)
	FILE *f = fopen(file_name,"a");
	for(int i = 0; i < _num_local_freq;i++){
		unsigned int counter = 0;
		for(int j = 0; j < _samples_per_data_set/_sk_step;j++){
			counter += rfi_data[i + _num_local_freq*j];
		}
		int freq_bin = slot_id + link_id*16 + 128*i;
		float freq_mhz = 800 - freq_bin*((float)400/1024);
		fprintf(f,"%f,%f\n",freq_mhz,(float)counter/_samples_per_data_set);
		INFO("RFI RECORDER: Frequency %f Percentage Masked: %f\n", freq_mhz , 100*(float)counter/_samples_per_data_set);
        }
	fclose(f);
	release_info_object(rfi_buf, buffer_ID);
        mark_buffer_empty(rfi_buf, unique_name.c_str(), buffer_ID);
        buffer_ID = (buffer_ID + 1) % rfi_buf->num_buffers;
    }
    INFO("RFI RECORDER: Closing Rfi Recorder");
}
