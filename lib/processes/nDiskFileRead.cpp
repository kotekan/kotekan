#include "nDiskFileRead.hpp"
#include <random>
#include "errors.h"
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

using std::string;

nDiskFileRead::nDiskFileRead(Config& config, const string& unique_name,
                                bufferContainer &buffer_containter) :
    KotekanProcess(config, unique_name, buffer_containter,
                     std::bind(&nDiskFileRead::main_thread, this))
    {   
        buf = get_buffer("in_buf");
        disk_id = 0; //TODO: move to config.
}

nDiskFileRead::~nDiskFileRead() {

}

void nDiskFileRead::apply_config(uint64_t fpga_seq) {

}

void nDiskFileRead::main_thread() {
    int num_disks = config.get_int(unique_name,"num_disks"); 
    string drive = config.get_string(unique_name,"drive");
    string capture = config.get_string(unique_name,"capture");
    unsigned int file_index = disk_id;
    int buf_id = disk_id;

    for (;;) {

        wait_for_empty_buffer(buf, buf_id);

        unsigned char* buf_ptr = buf->data[buf_id];
		char file_name[100];
		//snprintf(file_name, sizeof(file_name), "/drives/A/%d/20170426T110023Z_ARO_raw/%07d.vdif", disk_id,file_index);
    	snprintf(file_name, sizeof(file_name), "/mnt/%s/%d/%s/%07d.vdif",drive.c_str(), disk_id,capture.c_str(),file_index);
		FILE * in_file = fopen(file_name, "r");
		fseek(in_file, 0L, SEEK_END);
		long sz = ftell(in_file);
		rewind(in_file);
		assert(sz == buf->buffer_size); 
		
		fread(buf_ptr,buf->buffer_size,1,in_file);
		file_index += num_disks;

		fclose(in_file);

		set_data_ID(buf, buf_id, file_index);
        mark_buffer_full(buf, buf_id);
        buf_id = (buf_id + num_disks) % buf->num_buffers;

        INFO("nDiskFileRead: read %s\n", file_name);

    }
    mark_producer_done(buf, 0);
}
