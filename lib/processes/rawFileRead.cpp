#include "rawFileRead.hpp"
#include "errors.h"
#include "util.h"

inline bool file_exists(char * name) {
    struct stat buf;
    return (stat (name, &buf) == 0);
}

rawFileRead::rawFileRead(Config& config, const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rawFileRead::main_thread, this)) {

    buf = get_buffer("out_buf");
    register_producer(buf, unique_name.c_str());
    generate_info_object = config.get_bool(unique_name, "generate_info_object");
    repeat_frame = config.get_bool(unique_name, "repeat_frame");
    base_dir = config.get_string(unique_name, "base_dir");
    file_name = config.get_string(unique_name, "file_name");
    file_ext = config.get_string(unique_name, "file_ext");

    tmp_buf = nullptr;
    if (repeat_frame) {
        tmp_buf = malloc(buf->buffer_size);
        assert(tmp_buf != nullptr);
    }
}

rawFileRead::~rawFileRead() {
    if (tmp_buf != nullptr) {
        free(tmp_buf);
    }
}

void rawFileRead::apply_config(uint64_t fpga_seq) {
}

void rawFileRead::main_thread() {

    int file_num = 0;
    int buffer_id = 0;

    while(!stop_thread) {

        const int full_path_len = 200;
        char full_path[full_path_len];

        snprintf(full_path, full_path_len, "%s/%s_%07d.%s",
                base_dir.c_str(),
                file_name.c_str(),
                file_num,
                file_ext.c_str());

        if (!repeat_frame && !file_exists(full_path)) {
            INFO("rawFileRead: No file named %s, exiting read thread.", full_path);
            break;
        }

        // Get an empty buffer to write into
        wait_for_empty_frame(buf, unique_name.c_str(), buffer_id);

        if (repeat_frame) {
            if (file_num == 0) {
                INFO("Reading from file %s", full_path);
                FILE * fp = fopen(full_path, "rb");
                int bytes_read = fread(tmp_buf, sizeof (char), buf->buffer_size, fp);

                if (bytes_read != buf->buffer_size) {
                    ERROR("rawFileRead: Failed to read file %s!", full_path);
                    break;
                }

                INFO("rawFileRead: read data from %s, repeating data this in each frame.", full_path);
            }
            memcpy((void *)buf->data[buffer_id], tmp_buf, buf->buffer_size);
        } else {
            FILE * fp = fopen(full_path, "rb");
            int bytes_read = fread((void *)buf->data[buffer_id], sizeof (char), buf->buffer_size, fp);

            if (bytes_read != buf->buffer_size) {
                ERROR("rawFileRead: Failed to read file %s!", full_path);
                break;
            }

            INFO("rawFileRead: read data from %s", full_path);
        }
        //if (generate_info_object)
        //    hex_dump(8, (void *)buf.data[buffer_id], 8096);

        // The details don't matter for most subsystems
        if (generate_info_object) {
            set_data_ID(buf, buffer_id, 0);
            set_fpga_seq_num(buf, buffer_id, 0);
            set_stream_ID(buf, buffer_id, 0);
        }

        INFO("rawFileRead: marking buffer %s[%d] as full", buf->buffer_name, buffer_id);
        mark_frame_full(buf, unique_name.c_str(), buffer_id);

        file_num++;
        buffer_id = (buffer_id + 1) % buf->num_buffers;
    }

}
