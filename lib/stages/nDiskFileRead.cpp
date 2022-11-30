#include "nDiskFileRead.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_full, register_producer, wait_for_empt...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, ERROR

#include <algorithm>   // for max
#include <assert.h>    // for assert
#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <memory>      // for allocator_traits<>::value_type
#include <pthread.h>   // for pthread_setaffinity_np
#include <regex>       // for match_results<>::_Base_type
#include <sched.h>     // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdio.h>     // for fclose, fopen, fread, fseek, ftell, rewind, snprintf, FILE
#include <sys/types.h> // for uint


using std::string;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(nDiskFileRead);

nDiskFileRead::nDiskFileRead(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_containter) :
    Stage(config, unique_name, buffer_containter, std::bind(&nDiskFileRead::main_thread, this)) {
    // Get variables from config
    buf = get_buffer("out_buf"); // Buffer

    // Data paramters
    num_disks = config.get<uint32_t>(unique_name, "num_disks");

    // Data location parameters
    disk_base = config.get<std::string>(unique_name, "disk_base");
    disk_set = config.get<std::string>(unique_name, "disk_set");
    capture = config.get<std::string>(unique_name, "capture");
    starting_index = config.get<uint32_t>(unique_name, "starting_file_index");

    // Mark as producer
    register_producer(buf, unique_name.c_str());
}

void nDiskFileRead::main_thread() {

    // Create the threads
    file_thread_handles.resize(num_disks);
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i] = std::thread(&nDiskFileRead::file_read_thread, this, i);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        INFO("Setting thread affinity");
        for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
            CPU_SET(i, &cpuset);

        pthread_setaffinity_np(file_thread_handles[i].native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // Join the threads
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i].join();
    }
}

void nDiskFileRead::file_read_thread(int disk_id) {

    unsigned int buf_id = disk_id;
    // Starting File index
    unsigned int file_index = disk_id + starting_index;
    INFO("{:s}{:s}/{:d}/{:s}/{:07d}.vdif", disk_base, disk_set, disk_id, capture, file_index)
    // Endless loop
    while (!stop_thread) {

        unsigned char* buf_ptr =
            (unsigned char*)wait_for_empty_frame(buf, unique_name.c_str(), buf_id);
        if (buf_ptr == nullptr)
            break;

        char file_name[100]; // Find current file
        snprintf(file_name, sizeof(file_name), "%s%s/%d/%s/%07d.vdif", disk_base.c_str(),
                 disk_set.c_str(), disk_id, capture.c_str(), file_index);

        // Open current file for reading
        FILE* in_file = fopen(file_name, "r");

        // Make Sure file is the right size
        fseek(in_file, 0L, SEEK_END);
        long sz = ftell(in_file);
        rewind(in_file);

        if ((size_t)sz != buf->frame_size) {
            ERROR("File size {:d} Frame Size {:d}", sz, buf->frame_size);
        }
        assert((size_t)sz == buf->frame_size);

        // Read into buffer
        if (buf->frame_size != fread(buf_ptr, 1, sz, in_file)) {
            ERROR("Error reading from file!");
        }
        fclose(in_file);
        INFO("{:s} Read Complete Marking Frame ID {:d} Full\n", file_name, buf_id);

        mark_frame_full(buf, unique_name.c_str(), buf_id);

        // Go to next file
        buf_id = (buf_id + num_disks) % buf->num_frames;
        file_index += num_disks;
    }
}
