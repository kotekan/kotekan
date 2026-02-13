#include "mergedRawFileRead.hpp"

#include "BeamMetadata.hpp"    // for BeamMetadata
#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, allocate_new_metadata_object, get_metadata_container
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for atomic_add_lost_timesamples, zero_lost_samples
#include "kotekanLogging.hpp"  // for ERROR, INFO, FATAL_ERROR
#include "metadata.h"          // for metadataContainer

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstdio>     // for fread, fclose, fopen, snprintf, FILE
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t, uint8_t
#include <string.h>   // for strerror
#include <sys/stat.h> // for stat
#include <unistd.h>   // for sleep
#include <vector>     // for vector


inline bool file_exists(char* name) {
    struct stat buf;
    return (stat(name, &buf) == 0);
}

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(mergedRawFileRead);

mergedRawFileRead::mergedRawFileRead(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&mergedRawFileRead::main_thread, this)) {
    merged_buf = get_buffer("merged_buf");
    register_producer(merged_buf, unique_name.c_str());
    base_dir = config.get<std::string>(unique_name, "base_dir");
    file_name = config.get<std::string>(unique_name, "file_name");
    file_ext = config.get<std::string>(unique_name, "file_ext");
    prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", false);

    // Interrupt Kotekan if run out of files to read.
    end_interrupt = config.get_default<bool>(unique_name, "end_interrupt", false);
}

mergedRawFileRead::~mergedRawFileRead() {}

void mergedRawFileRead::main_thread() {

    int file_num = 0;
    int merged_frame_id = 0;
    uint8_t* merged_frame = nullptr;
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
                INFO("mergedRawFileRead: No file named {:s}, exiting read thread.", full_path);
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
            ERROR("mergedRawFileRead: Failed to read file {:s} metadata size value, {:s}",
                  full_path, strerror(errno));
            break;
        }

        num_frames_per_file = fileSize / (metadata_size + merged_buf->frame_size);

        INFO("File size: {:d} bytes, no. of frames: {:d}", fileSize, num_frames_per_file);

        // Read each frame from the file and copy into the buffer.
        for (uint32_t i = 0; i < num_frames_per_file; i++) {
            // Get an empty merged buffer to write into
            merged_frame = wait_for_empty_frame(merged_buf, unique_name.c_str(), merged_frame_id);
            if (merged_frame == nullptr)
                break;

            out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
            if (out_frame == nullptr)
                break;

            // If metadata exists then lets read it in.
            if (metadata_size != 0) {
                allocate_new_metadata_object(merged_buf, merged_frame_id);
                struct metadataContainer* mc = get_metadata_container(merged_buf, merged_frame_id);
                assert(metadata_size == mc->metadata_size);
                if (fread(mc->metadata, metadata_size, 1, fp) != 1) {
                    ERROR("mergedRawFileRead: Failed to read file {:s} metadata,", full_path);
                    break;
                }
                INFO("mergedRawFileRead: Read in metadata from file {:s}", full_path);
            }
            // Read one merged frames
            int bytes_read = fread((void*)merged_frame, sizeof(char), merged_buf->frame_size, fp);

            if (bytes_read != merged_buf->frame_size) {
                ERROR("mergedRawFileRea: Failed to read file {:s}!", full_path);
                break;
            }

            INFO("mergedRawFileRead: Read frame data from {:s} into {:s}[{:d}]", full_path,
                 merged_buf->buffer_name, merged_frame_id);
            // copy data from merged buffer to out buffer
            for (uint32_t i = 0; i < raw_frames_per_merged_frame; i++) {
                // Get the part for each frame and copy to the non merged buffer.
                FreqIDBeamMetadata* sub_frame_metadata =
                    (FreqIDBeamMetadata*)&merged_frame[out_frame_metadata_pos];
                out_frame_id++;
                mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
            }

            merged_frame_id = (merged_frame_id + 1) % merged_buf->num_frames;
        }
        fclose(fp);
        file_num++;
    }
}
