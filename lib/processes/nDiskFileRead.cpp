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
    buf = get_buffer("out_buf");
    num_disks = config.get_int(unique_name,"num_disks"); 
    disk_set = config.get_string(unique_name,"disk_set");
    capture = config.get_string(unique_name,"capture");
}

nDiskFileRead::~nDiskFileRead() {

}

void nDiskFileRead::apply_config(uint64_t fpga_seq) {

}

void nDiskFileRead::main_thread() {
    // Create the threads
    file_thread_handles.resize(num_disks);
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i] = std::thread(&nDiskFileRead::file_read_thread, this, i);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        INFO("Setting thread affinity");
        for (auto &i : config.get_int_array(unique_name, "cpu_affinity"))
            CPU_SET(i, &cpuset);

        pthread_setaffinity_np(file_thread_handles[i].native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // Join the threads
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i].join();
    }
    mark_producer_done(buf, 0);
}

void nDiskFileRead::file_read_thread(int disk_id) {
    int buf_id = disk_id;
    unsigned int file_index = disk_id;
    for (;;) {
        wait_for_empty_buffer(buf, buf_id);

        unsigned char* buf_ptr = buf->data[buf_id];
        char file_name[100];
        //snprintf(file_name, sizeof(file_name), "/drives/A/%d/20170426T110023Z_ARO_raw/%07d.vdif", disk_id,file_index);
        snprintf(file_name, sizeof(file_name), "/drives/%s/%d/%s/%07d.vdif",disk_set.c_str(), disk_id,capture.c_str(),file_index);
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

}
