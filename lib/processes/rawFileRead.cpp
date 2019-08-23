#include "rawFileRead.hpp"

#include "errors.h"
#include "util.h"

#include <csignal>
#include <errno.h>

inline bool file_exists(char* name) {
    struct stat buf;
    return (stat(name, &buf) == 0);
}

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rawFileRead);

rawFileRead::rawFileRead(Config& config, const string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rawFileRead::main_thread, this)) {

    buf = get_buffer("buf");
    register_producer(buf, unique_name.c_str());
    base_dir = config.get<std::string>(unique_name, "base_dir");
    file_name = config.get<std::string>(unique_name, "file_name");
    file_ext = config.get<std::string>(unique_name, "file_ext");

    // Interrupt Kotekan if run out of files to read.
    end_interrupt = config.get_default<bool>(unique_name, "end_interrupt", false);
}

rawFileRead::~rawFileRead() {}

void rawFileRead::main_thread() {

    int file_num = 0;
    int frame_id = 0;
    uint8_t* frame = NULL;

    while (!stop_thread) {

        const int full_path_len = 200;
        char full_path[full_path_len];

        snprintf(full_path, full_path_len, "%s/%s_%07d.%s", base_dir.c_str(), file_name.c_str(),
                 file_num, file_ext.c_str());

        if (!file_exists(full_path)) {
            // Interrupt Kotekan if run out of files to read.
            if (end_interrupt) {
                sleep(1);
                FATAL_ERROR("No more files to read. Shutting down Kotekan.");
                break;
            } else {
                INFO("rawFileRead: No file named %s, exiting read thread.", full_path);
                break;
            }
        }

        // Get an empty buffer to write into
        frame = wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL)
            break;

        FILE* fp = fopen(full_path, "rb");

        uint32_t metadata_size;

        if (fread((void*)&metadata_size, sizeof(uint32_t), 1, fp) != 1) {
            ERROR("rawFileRead: Failed to read file %s metadata size value, %s", full_path,
                  strerror(errno));
            break;
        }

        // If metadata exists then lets read it in.
        if (metadata_size != 0) {
            allocate_new_metadata_object(buf, frame_id);
            struct metadataContainer* mc = get_metadata_container(buf, frame_id);
            assert(metadata_size == mc->metadata_size);
            if (fread(mc->metadata, metadata_size, 1, fp) != 1) {
                ERROR("rawFileRead: Failed to read file %s metadata,", full_path);
                break;
            }
            INFO("rawFileRead: Read in metadata from file %s", full_path);
        }

        int bytes_read = fread((void*)frame, sizeof(char), buf->frame_size, fp);

        if (bytes_read != buf->frame_size) {
            ERROR("rawFileRead: Failed to read file %s!", full_path);
            break;
        }

        fclose(fp);

        INFO("rawFileRead: Read frame data from %s into %s[%i]", full_path, buf->buffer_name,
             frame_id);
        mark_frame_full(buf, unique_name.c_str(), frame_id);

        file_num++;
        frame_id = (frame_id + 1) % buf->num_frames;
    }
}
