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
#include "buffer.h"            // for Buffer, get_metadata_container, mark_frame_empty, registe...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "metadata.h"          // for metadataContainer
#include "util.h"              // for cp

#include "fmt.hpp"      // for format, parse_nonnegative_int, fmt
#include "fmt/chrono.h" // for localtime

#include <algorithm>    // for max
#include <atomic>       // for atomic_bool
#include <cstdint>      // for int64_t
#include <ctime>        // for time, tm
#include <errno.h>      // for errno, EEXIST
#include <exception>    // for exception
#include <fcntl.h>      // for open, SEEK_END, SEEK_SET, O_CREAT, O_WRONLY
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <memory>       // for allocator_traits<>::value_type
#include <pthread.h>    // for pthread_setaffinity_np
#include <regex>        // for match_results<>::_Base_type
#include <sched.h>      // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>    // for runtime_error
#include <stdio.h>      // for fprintf, size_t, printf, fclose, fopen, perror, FILE
#include <stdlib.h>     // for exit, size_t
#include <sys/stat.h>   // for mkdir
#include <system_error> // for error_code
#include <thread>       // for thread
#include <unistd.h>     // for write, lseek, close, ssize_t

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

    // get the output file extension
    file_extension = get_file_extension();
}

std::string nDiskMultiFormatWriter::get_file_extension() {
    // go through a nested if-else to find the requested file_format
    if (file_format == "raw") {
        return "raw";
    } else {
        if (file_format == "VDIF") {
            return "vdif";
        }
    }

    FATAL_ERROR("Unknown requested file format: {:s}\n", file_format);
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

void nDiskMultiFormatWriter::make_folders() {
    for (uint32_t i = 0; i < num_disks; i++) {
        // get the folder name
        std::string folder = get_dataset_folder_name(i);

        // make the folder and check if for any errors
        if (mkdir(folder.c_str(), 0777) != -1) {
            continue;
        }

        // if the folder exists it should be removed first
        if (errno == EEXIST) {
            INFO("The folder: {:s}, already exists.\nPlease delete the dataset, or use another "
                 "name.\n",
                 folder);
        } else {
            INFO("Error creating dataset folder: {:s} \n", folder);
        }
        FATAL_ERROR("Error in creating dataset folder!");
    }
}

void nDiskMultiFormatWriter::main_thread() {

    // get the time when this stage starts working
    invocation_time = fmt::localtime(std::time(nullptr));

    // create the folders based on the given disk_path
    make_folders();

    // Create the file writer threads
    file_thread_handles.resize(num_disks);
    for (uint32_t i = 0; i < num_disks; i++) {
        // select the writer thread based on the requested file_format
        // go through a nested if-else to find the requested file_format
        if (file_format == "raw") {
            file_thread_handles[i] =
                std::thread(&nDiskMultiFormatWriter::raw_file_write_thread, this, i);
        } else {
            if (file_format == "vdif") {
                file_thread_handles[i] =
                    std::thread(&nDiskMultiFormatWriter::vdif_file_write_thread, this, i);
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

/// disk data writer with a check
void nDiskMultiFormatWriter::write_and_check(const int file_handle, void* data_ptr,
                                             ssize_t data_size, const std::string& parameter_name,
                                             const std::string& file_name) {
    if (write(file_handle, data_ptr, data_size) != data_size) {
        FATAL_ERROR("Failed to write the {:s} to disk for file {:s}", parameter_name, file_name);
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

    // thread infinite loop
    while (!stop_thread) {

        // This call is blocking and wait until a full frame is available
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        DEBUG("Got buffer id: {:d}, disk id {:d}", frame_id, disk_id);

        // Check if the producer has finished, and we should exit.
        if (frame_id == -1) {
            break;
        }

        // Open a file to write the frame
        if (fd < 0) {
            // make the output file name
            file_name = get_dataset_folder_name(disk_id) + fmt::sprintf("/%010zu.", file_num)
                        + file_extension;
            // open a file and get its handle
            fd = open(file_name.c_str(), O_WRONLY | O_CREAT, 0666);

            // set num_frames to -1 just to write -1 at the beginning of the file
            num_frames = -1;
            // write the number of frames in this file
            write_and_check(fd, &num_frames, sizeof(num_frames), "number of frames", file_name);
            // reset the num_frames
            num_frames = 0;
        }

        // first check if the file has been opened to write
        if (fd == -1) {
            FATAL_ERROR("Cannot open file: {:s}", file_name);
        }

        // write the metadata and its size
        size_t metadata_size = 0;
        struct metadataContainer* mc = get_metadata_container(buf, frame_id);
        if (mc != nullptr) {
            metadata_size = mc->metadata_size;
        }
        // Write metadata size to disk, if there is no metadata in the frame, then
        // just save 0 to the first word.
        write_and_check(fd, &metadata_size, sizeof(metadata_size), "metadata_size", file_name);

        // if the metadata container is not null, write it as well
        if (mc != nullptr) {
            write_and_check(fd, mc->metadata, mc->metadata_size, "metadata", file_name);
        }

        // write the number of frequencies
        write_and_check(fd, &num_freq_per_output_frame, sizeof(num_freq_per_output_frame),
                        "number of frequencies", file_name);

        // write the frame size in bytes
        size_t frame_byte_size = (size_t)buf->frame_size;
        write_and_check(fd, &frame_byte_size, sizeof(size_t), "frame_size", file_name);

        // write the frame data finally
        write_and_check(fd, frame, buf->frame_size, "frame data", file_name);

        INFO("Data from buffer id {:d} written to {:s}", frame_id, file_name);

        // increment the number of frames written to the file
        num_frames++;

        // close the file and open a new one, if the number of frames written is equal to
        // max_frames_per_file
        if (num_frames == max_frames_per_file) {
            // special close: update the number of frames written to this file
            lseek(fd, 0, SEEK_SET);
            write_and_check(fd, &num_frames, sizeof(num_frames), "number of frames", file_name);
            lseek(fd, 0, SEEK_END);

            // now close the file handle
            if (close(fd) == -1) {
                ERROR("Cannot close the file {:s}", file_name);
            }
            // reset the file handle
            fd = -1;
            // increment the file index by the num_disks
            file_num += num_disks;
        }

        // TODO make release_info_object work for nConsumers.
        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        // get a new frame id
        frame_id = (frame_id + num_disks) % buf->num_frames;

    } // thread infinite loop

    // close any open file handle
    if (fd > 0) {
        // special close: update the number of frames written to this file
        lseek(fd, 0, SEEK_SET);
        write_and_check(fd, &num_frames, sizeof(num_frames), "number of frames", file_name);
        lseek(fd, 0, SEEK_END);

        // now close the file handle
        if (close(fd) == -1) {
            ERROR("Cannot close the file {:s}", file_name);
        }
    }
}

/// writes the incoming frames to an VDIF file
void nDiskMultiFormatWriter::vdif_file_write_thread(int /*disk_id*/) {
    INFO("Not Implemented yet!")
}
