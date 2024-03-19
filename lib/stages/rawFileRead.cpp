#include "rawFileRead.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, allocate_new_metadata_object, get_metadata_container
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for ERROR, INFO, FATAL_ERROR
#include "metadata.h"          // for metadataContainer

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstdio>     // for fread, fclose, fopen, snprintf, FILE
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t, uint8_t
#include <string.h>   // for strerror
#include <sys/stat.h> // for stat
#include <unistd.h>   // for sleep


inline bool file_exists(char* name) {
    struct stat buf;
    return (stat(name, &buf) == 0);
}

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rawFileRead);

rawFileRead::rawFileRead(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rawFileRead::main_thread, this)) {

    buf = get_buffer("buf");
    buf->register_producer(unique_name);
    base_dir = config.get<std::string>(unique_name, "base_dir");
    file_name = config.get<std::string>(unique_name, "file_name");
    file_ext = config.get<std::string>(unique_name, "file_ext");
    prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", false);

    // Interrupt Kotekan if run out of files to read.
    end_interrupt = config.get_default<bool>(unique_name, "end_interrupt", false);
}

rawFileRead::~rawFileRead() {}

void rawFileRead::main_thread() {

    int file_num = 0;
    int frame_id = 0;
    uint8_t* frame = nullptr;
    char hostname[64];
    gethostname(hostname, 64);

    while (!stop_thread) {

        const int full_path_len = 200;
        char full_path[full_path_len];

        if (prefix_hostname) {
            snprintf(full_path, full_path_len, "%s/%s_%s_%07d.%s", base_dir.c_str(), hostname,
                     file_name.c_str(), file_num, file_ext.c_str());
        } else {
            snprintf(full_path, full_path_len, "%s/%s_%07d.%s", base_dir.c_str(), file_name.c_str(),
                     file_num, file_ext.c_str());
        }

        INFO("Looking for file: {:s}", full_path);
        if (!file_exists(full_path)) {
            // Interrupt Kotekan if run out of files to read.
            if (end_interrupt) {
                sleep(1);
                FATAL_ERROR("No more files to read. Shutting down Kotekan.");
                break;
            } else {
                INFO("rawFileRead: No file named {:s}, exiting read thread.", full_path);
                break;
            }
        }

        FILE* fp = fopen(full_path, "rb");
        uint32_t metadata_size = 0;
        uint32_t fileSize, num_frames_per_file;

        // Work out the file size, metadata size and no. of frames per file.
        fseek(fp, 0, SEEK_END);
        fileSize = ftell(fp);
        rewind(fp);

        if (fread((void*)&metadata_size, sizeof(uint32_t), 1, fp) != 1) {
            ERROR("rawFileRead: Failed to read file {:s} metadata size value, {:s}", full_path,
                  strerror(errno));
            break;
        }

        num_frames_per_file = fileSize / (metadata_size + buf->frame_size);

        INFO("File size: {:d} bytes, no. of frames: {:d}", fileSize, num_frames_per_file);

        // Read each frame from the file and copy into the buffer.
        for (uint32_t i = 0; i < num_frames_per_file; i++) {

            // Get an empty buffer to write into
            frame = buf->wait_for_empty_frame(unique_name, frame_id);
            if (frame == nullptr)
                break;

            // If metadata exists then lets read it in.
            if (metadata_size != 0) {
                buf->allocate_new_metadata_object(frame_id);
                metadataContainer* mc = buf->get_metadata_container(frame_id);
                assert(metadata_size == mc->metadata_size);
                if (fread(mc->metadata, metadata_size, 1, fp) != 1) {
                    ERROR("rawFileRead: Failed to read file {:s} metadata,", full_path);
                    break;
                }
                INFO("rawFileRead: Read in metadata from file {:s}", full_path);
            }

            size_t bytes_read = fread((void*)frame, sizeof(char), buf->frame_size, fp);

            if (bytes_read != buf->frame_size) {
                ERROR("rawFileRead: Failed to read file {:s}!", full_path);
                break;
            }

            INFO("rawFileRead: Read frame data from {:s} into {:s}[{:d}]", full_path,
                 buf->buffer_name, frame_id);
            buf->mark_frame_full(unique_name, frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;
        }

        fclose(fp);
        file_num++;
    }
}
