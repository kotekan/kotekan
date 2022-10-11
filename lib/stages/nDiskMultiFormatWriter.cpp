// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
 * @file   nDiskMultiFormatWriter.cpp
 * @brief  This file implements nDiskMultiFormatWriter
 *         stage, multi-format multiple drives writer
 *
 * @author Mehdi Najafi
 * @date   12 SEP 2022
 *****************************************************/

#include "nDiskMultiFormatWriter.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, register_consumer, ...
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for get_lost_timesamples
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "util.h"              // for cp

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
#include <sys/stat.h> // for mkdir
#include <thread>     // for thread
#include <time.h>     // for gmtime, strftime, time, time_t
#include <unistd.h>   // for close, write, ssize_t

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(nDiskMultiFormatWriter);

nDiskMultiFormatWriter::nDiskMultiFormatWriter(Config& config, const string& unique_name,
                                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&nDiskMultiFormatWriter::main_thread, this)) {
    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());

    // Apply config.
    // retrieve the number of disk
    num_disks = config.get<uint32_t>(unique_name, "num_disks");
    // retrieve the path and file naming convention, like /drives/CHF/{n}/{DATE}T{TIME}Z_asm
    disk_path = config.get<std::string>(unique_name, "disk_path");
    // retrieve the requested output file format
    file_format = config.get<std::string>(unique_name, "file_format");
    // retrieve the expected number of frequencies in the output
    num_freq_per_output_frame = config.get<uint32_t>(unique_name, "num_freqs");
    // retrieve the maximum number of frames to be written to each file
    max_frames_per_file = config.get<int64_t>(unique_name, "max_frames_per_file");
    // write to disk boolean as a on/off switch for any real disk activity
    write_to_disk = config.get<bool>(unique_name, "write_to_disk");
    // write to disk boolean as a on/off switch for any real disk activity
    write_metadata_and_gains =
        config.get_default<bool>(unique_name, "write_metadata_and_gains", true);

    // get the output file extension
    file_extension = get_file_extension();
}

std::string nDiskMultiFormatWriter::get_file_extension() {
    // go through a nested if-else to find the requested file_format
    if (file_format == "raw") {
        return "raw";
    } else {
        if (file_format == "HDF5") {
            return "h5";
        }
    }

    ERROR("Unknown requested file format: {:s}\n", file_format);
    exit(-1);
    return "";
}

void nDiskMultiFormatWriter::string_replace_all(std::string& str, const std::string& from,
                                                const std::string& to) {
    if (from.empty())
        return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

std::string nDiskMultiFormatWriter::get_dataset_timestamp() {
    return fmt::format("{:%Y%m%d}T{:%H%M%S}Z", invocation_time, invocation_time);
}

std::string nDiskMultiFormatWriter::get_dataset_folder_name(const int n) {
    std::string path(disk_path);
    string_replace_all(path, "{DATE}", fmt::format("{:%Y%m%d}", invocation_time));
    string_replace_all(path, "{TIME}", fmt::format("{:%H%M%S}", invocation_time));
    string_replace_all(path, "{n}", fmt::format("{}", n));
    return path;
}

void nDiskMultiFormatWriter::save_metadata() {

    for (uint32_t i = 0; i < num_disks; i++) {

        std::string file_name = get_dataset_folder_name(i) + "/settings.txt";

        FILE* info_file = fopen(file_name.c_str(), "w");

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
        fprintf(info_file, "num_frames=%d\n", 1); // Always one
        fprintf(info_file, "num_timesamples=%d\n", samples_per_file);
        fprintf(info_file, "header_len=%d\n", vdif_header_len); // VDIF
        fprintf(info_file, "packet_len=%d\n",
                vdif_header_len + num_freq); // 1056 for VDIF with 1024 freq
        fprintf(info_file, "offset=%d\n", 0);
        fprintf(info_file, "data_bits=%d\n", bit_depth);
        fprintf(info_file, "stride=%d\n", 1);
        fprintf(info_file, "stream_id=n/a\n");
        fprintf(info_file, "note=\"%s\"\n", note.c_str());
        fprintf(info_file, "start_time=%s\n", get_dataset_timestamp().c_str());
        fprintf(info_file, "num_disks=%d\n", num_disks);
        fprintf(info_file, "disk_path=%s\n", get_dataset_folder_name(i).c_str());
        fprintf(info_file, "# Warning: The start time is when the program starts it, the time "
                           "recorded in the packets is more accurate.\n");

        fclose(info_file);

        INFO("Created metadata file: {:s}\n", file_name);
    }
}

void nDiskMultiFormatWriter::make_folders() {
    std::error_code ec;
    for (uint32_t i = 0; i < num_disks; i++) {
        std::string folder = get_dataset_folder_name(i);
        int err = mkdir(folder.c_str(), 0777);

        if (err != -1) {
            continue;
        }

        if (errno == EEXIST) {
            printf("The folder: %s, already exists.\nPlease delete the data set, or use another "
                   "name.\n",
                   folder.c_str());
        } else {
            perror("Error creating data set directory.\n");
            printf("The directory was: %s \n", folder.c_str());
        }
        exit(errno);
    }
}

void nDiskMultiFormatWriter::main_thread() {

    // get the time
    invocation_time = fmt::localtime(std::time(nullptr));

    // create the folders
    if (write_to_disk) {
        make_folders();
    }

    if (write_to_disk && write_metadata_and_gains) {

        // Copy gain files
        std::vector<std::string> gain_files =
            config.get<std::vector<std::string>>(unique_name, "gain_files");
        for (uint32_t i = 0; i < num_disks; i++) {
            for (uint32_t j = 0; j < gain_files.size(); ++j) {
                unsigned int last_slash_pos = gain_files[j].find_last_of("/\\");
                std::string dest = fmt::format(fmt("{:s}/{:s}"), get_dataset_folder_name(i),
                                               gain_files[j].substr(last_slash_pos + 1));
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
        save_metadata();
    }

    // Create the file writer threads
    file_thread_handles.resize(num_disks);
    for (uint32_t i = 0; i < num_disks; i++) {
        // select the writer thread based on the requested file_format
        // go through a nested if-else to find the requested file_format
        if (file_format == "raw") {
            file_thread_handles[i] =
                std::thread(&nDiskMultiFormatWriter::raw_file_write_thread, this, i);
        } else {
            if (file_format == "HDF5") {
                file_thread_handles[i] =
                    std::thread(&nDiskMultiFormatWriter::hdf5_file_write_thread, this, i);
            }
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        INFO("Setting thread affinity");
        for (auto& j : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
            CPU_SET(j, &cpuset);

        pthread_setaffinity_np(file_thread_handles[i].native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // Join the threads
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i].join();
    }
}

/// writes the incoming frames to a raw file
void nDiskMultiFormatWriter::raw_file_write_thread(int disk_id) {
    int fd = -1;
    size_t file_num = disk_id;
    int frame_id = disk_id;
    uint8_t* frame = nullptr;
    int64_t num_frames = 0;
    std::string file_name;
    ssize_t bytes_written;

    // thread infinite loop
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

        // Open the file to write
        if (write_to_disk) {

            if (fd < 0) {
                // make the output file name
                file_name = get_dataset_folder_name(disk_id) + fmt::sprintf("%010zu.", file_num)
                            + file_extension;
                // open a file and get its handle
                fd = open(file_name.c_str(), O_WRONLY | O_CREAT, 0666);

                // set num_frames to -1 just to write -1 at the beginning of the file
                num_frames = -1;
                // write the number of frames in this file
                write(fd, &num_frames, sizeof(num_frames));
                // reset the num_frames
                num_frames = 0;
            }

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: {:s}", file_name);
                exit(errno);
            }

            // write the metadata and its size
            size_t metadata_size = 0;
            struct metadataContainer* mc = get_metadata_container(buf, frame_id);
            if (mc != nullptr) {
                metadata_size = mc->metadata_size;
            }
            // Write metadata size to disk, if there is no metadata in the frame, then
            // just save 0 to the first word.
            if (write(fd, (void*)&metadata_size, sizeof(metadata_size))
                != (int32_t)sizeof(metadata_size)) {
                ERROR("Failed to write metadata_size to disk for file {:s}", file_name);
                exit(-1);
            }
            if (mc != nullptr) {
                if (write(fd, mc->metadata, mc->metadata_size) != (int32_t)mc->metadata_size) {
                    ERROR("Failed to write metadata_size to disk for file {:s}", file_name);
                    exit(-1);
                }
            }

            // write the number of frequencies
            bytes_written =
                write(fd, &num_freq_per_output_frame, sizeof(num_freq_per_output_frame));

            // write the frame size in bytes
            size_t frame_byte_size = (size_t)buf->frame_size;
            bytes_written = write(fd, &frame_byte_size, sizeof(size_t));

            // write the frame data
            bytes_written = write(fd, frame, buf->frame_size);

            if (bytes_written != buf->frame_size) {
                ERROR("Failed to write buffer to disk!!!  Abort, Panic, etc.");
                exit(-1);
            } else {
                // INFO("Data written to file!");
            }

            INFO("Data written to {:s}, lost_packets {:d}", file_name,
                 get_lost_timesamples(buf, frame_id));

            // increment the number of frames written to the file
            num_frames++;

            // close the file and open a new one, if the number of frames written is equal to
            // max_frames_per_file
            if (num_frames == max_frames_per_file) {
                // special close
                // update the number of frames written to this file
                lseek(fd, 0, SEEK_SET);
                write(fd, &num_frames, sizeof(num_frames));
                lseek(fd, 0, SEEK_END);

                // now close the file handle
                if (close(fd) == -1) {
                    ERROR("Cannot close file {:s}", file_name);
                }
                // reset the file handle
                fd = -1;
                // increment the file index by the num_disks
                file_num += num_disks;
            }

        } else {
            // usleep(0.070 * 1e6);
            INFO("Disk id {:d}, Lost Packets {:d}, buffer id {:d}", disk_id,
                 get_lost_timesamples(buf, frame_id), frame_id);
        }

        // TODO make release_info_object work for nConsumers.
        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        // get a new frame id
        frame_id = (frame_id + num_disks) % buf->num_frames;

    } // thread infinite loop


    // close any open file handle
    if (fd > 0) {
        // special close
        // update the number of frames written to this file
        lseek(fd, 0, SEEK_SET);
        write(fd, &num_frames, sizeof(num_frames));
        lseek(fd, 0, SEEK_END);

        // now close the file handle
        if (close(fd) == -1) {
            ERROR("Cannot close file {:s}", file_name);
        }
    }
}

/// writes the incoming frames to an HDF5 file
void nDiskMultiFormatWriter::hdf5_file_write_thread(int /*disk_id*/) {
    INFO("Not Implemented yet!")
}
