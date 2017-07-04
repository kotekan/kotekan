#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <functional>
#include <thread>
#include <sys/stat.h>
#include "nDiskFileWrite.hpp"
#include "buffers.h"
#include "errors.h"
#include "util.h"

nDiskFileWrite::nDiskFileWrite(Config& config, const string& unique_name,
                                bufferContainer &buffer_containter) :
        KotekanProcess(config, unique_name, buffer_containter,
                       std::bind(&nDiskFileWrite::main_thread, this))
{
    buf = get_buffer("in_buf");
    apply_config(0);
}

nDiskFileWrite::~nDiskFileWrite() {
}

void nDiskFileWrite::apply_config(uint64_t fpga_seq) {
    disk_base = config.get_string(unique_name, "disk_base");
    num_disks = config.get_int(unique_name, "num_disks");
    disk_set = config.get_string(unique_name, "disk_set");
    write_to_disk = config.get_bool(unique_name, "write_to_disk");
    instrument_name = config.get_string(unique_name, "instrument_name");
}

void nDiskFileWrite::save_meta_data() {

    char file_name[200];
    snprintf(file_name, sizeof(file_name), "%s/%s/%d/%s/settings.txt",
                    disk_base.c_str(), disk_set.c_str(), disk_id, dataset_name.c_str());

    FILE * info_file = fopen(file_name, "w");

    if(!info_file) {
        ERROR("Error creating info file: %s\n", file_name);
        exit(-1);
    }

    const int data_format_version = 3;
    int num_freq = config.get_int(unique_name,"num_freq");
    int num_elements = config.get_int(unique_name,"num_elements");
    int samples_per_file = config.get_int(unique_name,"samples_per_data_set");
    const int vdif_header_len = 32;
    const int bit_depth = 4;
    string note = config.get_string(unique_name,"note");

    fprintf(info_file, "format_version_number=%02d\n", data_format_version);
    fprintf(info_file, "num_freq=%d\n", num_freq);
    fprintf(info_file, "num_inputs=%d\n", num_elements);
    fprintf(info_file, "num_frames=%d\n", 1); // Always one for this VDIF
    fprintf(info_file, "num_timesamples=%d\n", samples_per_file);
    fprintf(info_file, "header_len=%d\n", vdif_header_len); // VDIF
    fprintf(info_file, "packet_len=%d\n", vdif_header_len + num_freq); // 1056 for VDIF with 1024 freq
    fprintf(info_file, "offset=%d\n", 0);
    fprintf(info_file, "data_bits=%d\n", bit_depth);
    fprintf(info_file, "stride=%d\n", 1);
    fprintf(info_file, "stream_id=n/a\n");
    fprintf(info_file, "note=\"%s\"\n", note.c_str());
    fprintf(info_file, "start_time=%s\n", dataset_name.c_str());
    fprintf(info_file, "num_disks=%d\n", num_disks);
    fprintf(info_file, "disk_set=%s\n", disk_set.c_str());
    fprintf(info_file, "# Warning: The start time is when the program starts it, the time recorded in the packets is more accurate\n");

    fclose(info_file);

    INFO("Created meta data file: %s\n", file_name);
}

// TODO instead of there being N disks of this tread started, this thread should
// start N threads to write the data.
void nDiskFileWrite::main_thread() {

    // TODO This is a very C style, maybe make it more C++11 like?
    char data_time[64];
    char data_set_c[150];
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = gmtime(&rawtime);
    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(data_set_c, sizeof(data_set_c), "%s_%s_raw", data_time, instrument_name.c_str());
    dataset_name = data_set_c;

    if (write_to_disk) {
        make_raw_dirs(disk_base.c_str(), disk_set.c_str(), dataset_name.c_str(), num_disks);
        // Copy gain files
        std::vector<std::string> gain_files = config.get_string_array(unique_name, "gain_files");
        for (uint32_t i = 0; i < num_disks; ++i) {
            for (uint32_t j = 0; j < gain_files.size(); ++j) {
                unsigned int last_slash_pos = gain_files[j].find_last_of("/\\");
                std::string dest = disk_base + "/" + disk_set + std::to_string(i) + "/" +
                        dataset_name + "/" +
                        gain_files[j].substr(last_slash_pos+1);
                // Copy the gain file
                cp(dest.c_str(), gain_files[j].c_str());

            }
        }
INFO("OK");
    }

    // Create the threads
    file_thread_handles.resize(num_disks);
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i] = std::thread(&nDiskFileWrite::file_write_thread, this, i);

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
}

void nDiskFileWrite::file_write_thread(int disk_id) {

    int fd;
    int file_num = disk_id;
    int buffer_id = disk_id;

    string gain_file_dir = config.get_string(unique_name,"gain_file_dir");

    sleep(1);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    INFO("Setting thread affinity");
    for (int j = 10; j < 12; j++) CPU_SET(j, &cpuset);
//    CPU_SET(11-(disk_id%2),&cpuset);
    pthread_setaffinity_np(this_thread.native_handle(), sizeof(cpu_set_t), &cpuset);

    for (;;) {

        // This call is blocking.
        buffer_id = get_full_buffer_from_list(buf, &buffer_id, 1);

        //INFO("Got buffer id: %d, disk id %d", buffer_id, disk_id);

        // Check if the producer has finished, and we should exit.
        if (buffer_id == -1) {
            break;
        }

        const int file_name_len = 100;
        char file_name[file_name_len];

        snprintf(file_name, file_name_len, "%s/%s/%d/%s/%07d.vdif",
                disk_base.c_str(),
                disk_set.c_str(),
                disk_id,
                dataset_name.c_str(),
                file_num);

        struct ErrorMatrix * error_matrix = get_error_matrix(buf, buffer_id);

        // Open the file to write
        if (write_to_disk) {

            fd = open(file_name, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: %s", file_name);
                exit(errno);
            }

            ssize_t bytes_writen = write(fd, buf->data[buffer_id], buf->buffer_size);

            if (bytes_writen != buf->buffer_size) {
                ERROR("Failed to write buffer to disk!!!  Abort, Panic, etc.");
                exit(-1);
            } else {
                 //INFO("Data writen to file!");
            }

            if (close(fd) == -1) {
                ERROR("Cannot close file %s", file_name);
            }

            INFO("Data file write done for %s, lost_packets %d", file_name, error_matrix->bad_timesamples);
        } else {
            //usleep(0.070 * 1e6);
            INFO("Disk id %d, Lost Packets %d, buffer id %d", disk_id, error_matrix->bad_timesamples, buffer_id );
        }

        // Zero the buffer (needed for VDIF packet processing)
        zero_buffer(buf, buffer_id);

        // TODO make release_info_object work for nConsumers.
        release_info_object(buf, buffer_id);
        mark_buffer_empty(buf, buffer_id);

        buffer_id = ( buffer_id + num_disks ) % buf->num_buffers;
        file_num += num_disks;
    }
}
