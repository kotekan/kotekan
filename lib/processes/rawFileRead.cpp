#include "rawFileRead.hpp"
#include "errors.h"
#include "util.h"
#include <errno.h>

inline bool file_exists(char * name) {
    struct stat buf;
    return (stat (name, &buf) == 0);
}

REGISTER_KOTEKAN_PROCESS(rawFileRead);

rawFileRead::rawFileRead(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rawFileRead::main_thread, this)) {

    buf = get_buffer("buf");
    register_producer(buf, unique_name.c_str());
    base_dir = config.get_string(unique_name, "base_dir");
    file_name = config.get_string(unique_name, "file_name");
    file_ext = config.get_string(unique_name, "file_ext");
}

rawFileRead::~rawFileRead() {
}

void rawFileRead::apply_config(uint64_t fpga_seq) {
}

void rawFileRead::main_thread() {

    int file_num = 0;
    int frame_id = 0;
    uint8_t * frame = NULL;

    while(!stop_thread) {

        const int full_path_len = 200;
        char full_path[full_path_len];

        snprintf(full_path, full_path_len, "%s/%s_%07d.%s",
                base_dir.c_str(),
                file_name.c_str(),
                file_num,
                file_ext.c_str());

        if (!file_exists(full_path)) {
            INFO("rawFileRead: No file named %s, exiting read thread.", full_path);
            break;
        }

        // Get an empty buffer to write into
        frame = wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        FILE * fp = fopen(full_path, "rb");

        uint32_t metadata_size;

        if (fread((void *)&metadata_size, sizeof(uint32_t), 1, fp) != 1) {
            ERROR("rawFileRead: Failed to read file %s metadata size value, %s", full_path, strerror(errno));
            break;
        }

        // If metadata exists then lets read it in.
        if (metadata_size != 0) {
            allocate_new_metadata_object(buf, frame_id);
            struct metadataContainer * mc = get_metadata_container(buf, frame_id);
            // assert(metadata_size <= mc->metadata_size);
            assert(metadata_size <= mc->metadata_size);
            memset(mc + metadata_size, 0, mc->metadata_size - metadata_size);

            if (fread(mc->metadata, metadata_size, 1, fp) != 1) {
                ERROR("rawFileRead: Failed to read file %s metadata,", full_path);
                break;
            }
            INFO("rawFileRead: Read in metadata from file %s", full_path);
        }

        int bytes_read = fread((void *)frame, sizeof(char), buf->frame_size, fp);

        if (bytes_read != buf->frame_size) {
            ERROR("rawFileRead: Failed to read file %s!", full_path);
            break;
        }


        INFO("rawFileRead: Read frame data from %s into %s[%i]", full_path, buf->buffer_name, frame_id);
        mark_frame_full(buf, unique_name.c_str(), frame_id);

        file_num++;
        frame_id = (frame_id + 1) % buf->num_frames;
    }
}
