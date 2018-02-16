#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <time.h>
#include <thread>
#include "bufferStatus.hpp"
#include "buffer.h"
#include "errors.h"
#include "output_formating.h"

REGISTER_KOTEKAN_PROCESS(bufferStatus);

bufferStatus::bufferStatus(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&bufferStatus::main_thread, this)){
    buffers = buffer_container.get_buffer_map();
    //buf = get_buffer("in_buf");
    //register_consumer(buf, unique_name.c_str());
}

bufferStatus::~bufferStatus() {
}

void bufferStatus::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
    time_delay = config.get_int(unique_name,"time_delay");
}

void bufferStatus::main_thread() {
    apply_config(0);

    // Wait for, and drop full buffers
    while (!stop_thread) {
	usleep(time_delay);
	map<string, Buffer*>::iterator it;
	INFO("BUFFER_STATUS");
	for ( it = buffers.begin(); it != buffers.end(); it++ )
	{
		print_buffer_status(it->second);
	}
    }
    INFO("Closing Buffer Status thread");
}
