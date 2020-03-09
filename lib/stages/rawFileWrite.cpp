#include "rawFileWrite.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for Buffer, get_metadata_container, mark_frame_empty, regis...
#include "bufferContainer.hpp"   // for bufferContainer
#include "kotekanLogging.hpp"    // for ERROR, INFO
#include "metadata.h"            // for metadataContainer
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "visUtil.hpp"           // for current_time

#include <atomic>     // for atomic_bool
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fcntl.h>    // for open, O_CREAT, O_WRONLY
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint32_t, int32_t, uint8_t
#include <stdio.h>    // for snprintf
#include <stdlib.h>   // for exit
#include <unistd.h>   // for write, close, gethostname, ssize_t
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(rawFileWrite);

rawFileWrite::rawFileWrite(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rawFileWrite::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    _base_dir = config.get<std::string>(unique_name, "base_dir");
    _file_name = config.get<std::string>(unique_name, "file_name");
    _file_ext = config.get<std::string>(unique_name, "file_ext");
    _num_frames_per_file = config.get_default<uint32_t>(unique_name, "num_frames_per_file", 1);
    _prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
}

rawFileWrite::~rawFileWrite() {}

void rawFileWrite::main_thread() {

    int fd;
    int file_num = 0;
    int frame_id = 0;
    uint32_t frame_ctr = 0;
    uint8_t* frame = nullptr;
    char hostname[64];
    gethostname(hostname, 64);
    bool isFileOpen = false;

    const int full_path_len = 200;
    char full_path[full_path_len];

    auto& write_time_metric =
        Metrics::instance().add_gauge("kotekan_rawfilewrite_write_time_seconds", unique_name);
    while (!stop_thread) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        // Start timing the write time
        double st = current_time();

        if (!isFileOpen) {

            if (_prefix_hostname) {
                snprintf(full_path, full_path_len, "%s/%s_%s_%07d.%s", _base_dir.c_str(), hostname,
                         _file_name.c_str(), file_num, _file_ext.c_str());
            } else {
                snprintf(full_path, full_path_len, "%s/%s_%07d.%s", _base_dir.c_str(),
                         _file_name.c_str(), file_num, _file_ext.c_str());
            }

            fd = open(full_path, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: {:s}", full_path);
                exit(errno);
            }

            isFileOpen = true;
        }

        // Write the meta data to disk
        uint32_t metadata_size = 0;
        struct metadataContainer* mc = get_metadata_container(buf, frame_id);
        if (mc != nullptr) {
            metadata_size = mc->metadata_size;
        }
        // Write metadata size to disk, if there is no metadata in the frame, then
        // just save 0 to the first word.
        if (write(fd, (void*)&metadata_size, sizeof(metadata_size))
            != (int32_t)sizeof(metadata_size)) {
            ERROR("Failed to write metadata_size to disk for file {:s}", full_path);
            exit(-1);
        }
        if (mc != nullptr) {
            if (write(fd, mc->metadata, mc->metadata_size) != (int32_t)mc->metadata_size) {
                ERROR("Failed to write metadata_size to disk for file {:s}", full_path);
                exit(-1);
            }
        }

        // Write the contents of the buffer frame to disk.
        ssize_t bytes_writen = write(fd, frame, buf->frame_size);

        if (bytes_writen != buf->frame_size) {
            ERROR("Failed to write buffer to disk for file {:s}", full_path);
            exit(-1);
        }

        INFO("Data file write done for {:s}", full_path);

        frame_ctr++;

        if (frame_ctr == _num_frames_per_file) {
            if (close(fd) == -1) {
                ERROR("Cannot close file {:s}", full_path);
            }
            isFileOpen = false;
            frame_ctr = 0;
            file_num++;
        }

        double elapsed = current_time() - st;
        write_time_metric.set(elapsed);

        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + 1) % buf->num_frames;
    }
}
