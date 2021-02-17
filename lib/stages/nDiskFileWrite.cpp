#include "nDiskFileWrite.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for get_lost_timesamples
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "util.h"              // for cp, make_raw_dirs

#include "fmt.hpp" // for format, fmt

#include <algorithm>  // for max
#include <atomic>     // for atomic_bool
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fcntl.h>    // for open, O_CREAT, O_WRONLY
#include <functional> // for _Bind_helper<>::type, bind, function
#include <memory>     // for allocator_traits<>::value_type
#include <pthread.h>  // for pthread_setaffinity_np
#include <regex>      // for match_results<>::_Base_type
#include <sched.h>    // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>  // for runtime_error
#include <stdio.h>    // for fprintf, snprintf, fclose, fopen, FILE, size_t
#include <stdlib.h>   // for exit
#include <thread>     // for thread
#include <time.h>     // for gmtime, strftime, time, time_t
#include <unistd.h>   // for close, write, ssize_t


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(nDiskFileWrite);

nDiskFileWrite::nDiskFileWrite(Config& config, const string& unique_name,
                               bufferContainer& buffer_containter) :
    Stage(config, unique_name, buffer_containter, std::bind(&nDiskFileWrite::main_thread, this)) {
    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());

    // Apply config.
    disk_base = config.get<std::string>(unique_name, "disk_base");
    num_disks = config.get<uint32_t>(unique_name, "num_disks");
    disk_set = config.get<std::string>(unique_name, "disk_set");
    file_ext = config.get_default<std::string>(unique_name, "file_ext", "vdif");
    write_to_disk = config.get<bool>(unique_name, "write_to_disk");
    instrument_name =
        config.get_default<std::string>(unique_name, "instrument_name", "no_name_set");
    write_metadata_and_gains =
        config.get_default<bool>(unique_name, "write_metadata_and_gains", true);
    print_lost_sample_number = 
	config.get_default<bool>(unique_name, "print_lost_sample_number", true);

}

nDiskFileWrite::~nDiskFileWrite() {}

void nDiskFileWrite::save_meta_data(char* timestr) {

    for (uint32_t i = 0; i < num_disks; ++i) {

        char file_name[200];
        snprintf(file_name, sizeof(file_name), "%s/%s/%d/%s/settings.txt", disk_base.c_str(),
                 disk_set.c_str(), i, dataset_name.c_str());

        FILE* info_file = fopen(file_name, "w");

        if (!info_file) {
            ERROR("Error creating info file: {:s}\n", file_name);
            exit(-1);
        }

        const int data_format_version = 3;
        int num_freq = config.get<int>(unique_name, "num_freq");
        int num_elements = config.get<int>(unique_name, "num_elements");
        int samples_per_file = config.get<int>(unique_name, "samples_per_data_set");
        const int vdif_header_len = 32;
        const int bit_depth = 4;
        string note = config.get<std::string>(unique_name, "note");

        fprintf(info_file, "format_version_number=%02d\n", data_format_version);
        fprintf(info_file, "num_freq=%d\n", num_freq);
        fprintf(info_file, "num_inputs=%d\n", num_elements);
        fprintf(info_file, "num_frames=%d\n", 1); // Always one for this VDIF
        fprintf(info_file, "num_timesamples=%d\n", samples_per_file);
        fprintf(info_file, "header_len=%d\n", vdif_header_len); // VDIF
        fprintf(info_file, "packet_len=%d\n",
                vdif_header_len + num_freq); // 1056 for VDIF with 1024 freq
        fprintf(info_file, "offset=%d\n", 0);
        fprintf(info_file, "data_bits=%d\n", bit_depth);
        fprintf(info_file, "stride=%d\n", 1);
        fprintf(info_file, "stream_id=n/a\n");
        fprintf(info_file, "note=\"%s\"\n", note.c_str());
        fprintf(info_file, "start_time=%s\n", timestr); // dataset_name.c_str());
        fprintf(info_file, "num_disks=%d\n", num_disks);
        fprintf(info_file, "disk_set=%s\n", disk_set.c_str());
        fprintf(info_file, "# Warning: The start time is when the program starts it, the time "
                           "recorded in the packets is more accurate\n");

        fclose(info_file);

        INFO("Created meta data file: {:s}\n", file_name);
    }
}


void nDiskFileWrite::main_thread() {

    // TODO This is a very C style, maybe make it more C++11 like?
    char data_time[64];
    char data_set_c[150];
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = gmtime(&rawtime);
    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(data_set_c, sizeof(data_set_c), "%s_%s_%s", data_time, instrument_name.c_str(), 
             file_ext.c_str());
    dataset_name = data_set_c;

    if (write_to_disk) {
        make_raw_dirs(disk_base.c_str(), disk_set.c_str(), dataset_name.c_str(), num_disks);
    }

    if (write_to_disk && write_metadata_and_gains) {

        // Copy gain files
        std::vector<std::string> gain_files =
            config.get<std::vector<std::string>>(unique_name, "gain_files");
        for (uint32_t i = 0; i < num_disks; ++i) {
            for (uint32_t j = 0; j < gain_files.size(); ++j) {
                unsigned int last_slash_pos = gain_files[j].find_last_of("/\\");
                std::string dest =
                    fmt::format(fmt("{:s}/{:s}/{:d}/{:s}/{:s}"), disk_base, disk_set, i,
                                dataset_name, gain_files[j].substr(last_slash_pos + 1));
                // Copy the gain file
                if (cp(dest.c_str(), gain_files[j].c_str()) != 0) {
                    ERROR("Could not copy {:s} to {:s}\n", gain_files[j], dest);
                    exit(-1);
                } else {
                    INFO("Copied gains from {:s} to {:s}\n", gain_files[j], dest);
                }
            }
        }
        // save settings
        save_meta_data(data_time);
    }

    // Create the threads
    file_thread_handles.resize(num_disks);
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i] = std::thread(&nDiskFileWrite::file_write_thread, this, i);

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

void nDiskFileWrite::file_write_thread(int disk_id) {

    int fd;
    size_t file_num = disk_id;
    int frame_id = disk_id;
    uint8_t* frame = nullptr;

    while (!stop_thread) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        // INFO("Got buffer id: {:d}, disk id {:d}", frame_id, disk_id);

        // Check if the producer has finished, and we should exit.
        if (frame_id == -1) {
            break;
        }

        const int file_name_len = 100;
        char file_name[file_name_len];

        snprintf(file_name, file_name_len, "%s/%s/%d/%s/%010zu.%s", disk_base.c_str(),
                 disk_set.c_str(), disk_id, dataset_name.c_str(), file_num, file_ext.c_str());

        // Open the file to write
        if (write_to_disk) {

            fd = open(file_name, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: {:s}", file_name);
                exit(errno);
            }

            ssize_t bytes_writen = write(fd, frame, buf->frame_size);

            if (bytes_writen != buf->frame_size) {
                ERROR("Failed to write buffer to disk!!!  Abort, Panic, etc.");
                exit(-1);
            } else {
                // INFO("Data writen to file!");
            }

            if (close(fd) == -1) {
                ERROR("Cannot close file {:s}", file_name);
            }
            if (print_lost_sample_number){
               INFO("Data file write done for {:s}, lost_packets {:d}", file_name,
                    get_lost_timesamples(buf, frame_id));
            }
	    else{
	       INFO("Data file write done for {:s}", file_name);
	    }
        } else {
            // usleep(0.070 * 1e6);
	    if (print_lost_sample_number){
               INFO("Disk id {:d}, Lost Packets {:d}, buffer id {:d}", disk_id,
                    get_lost_timesamples(buf, frame_id), frame_id);
        
	    }
	    else{
	       INFO("Disk id {:d}", disk_id);	    
	    }
        }
        // TODO make release_info_object work for nConsumers.
        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + num_disks) % buf->num_frames;
        file_num += num_disks;
    }
}
