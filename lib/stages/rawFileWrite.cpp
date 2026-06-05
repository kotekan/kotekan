#include "rawFileWrite.hpp"

#include "Config.hpp"              // for Config
#include "NDArray.hpp"             // for GenericNDArray, Config
#include "StageFactory.hpp"        // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"              // for Buffer
#include "bufferContainer.hpp"     // for bufferContainer
#include "errors.h"                // for ReturnCode, exit_kotekan
#include "kotekanLogging.hpp"      // for ERROR, DEBUG, FATAL_ERROR, INFO
#include "metadata.hpp"            // for metadataObject
#include "prometheusMetrics.hpp"   // for Metrics, Gauge
#include "visUtil.hpp"             // for current_time
#include "waitingForMaxFrames.hpp" // for waiting_for_max_frames

#include "fmt.hpp" // for compile_string_to_view

#include <atomic>     // for __atomic_base, atomic
#include <errno.h>    // for errno
#include <fcntl.h>    // for open, O_CREAT, O_WRONLY
#include <fmt/core.h> // for format
#include <functional> // for bind, function
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <stdint.h>   // for uint32_t, int32_t, uint8_t
#include <stdio.h>    // for snprintf, size_t
#include <stdlib.h>   // for exit
#include <unistd.h>   // for write, close, gethostname, ssize_t


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(rawFileWrite);

rawFileWrite::rawFileWrite(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rawFileWrite::main_thread, this)) {

    buf = get_buffer("in_buf");
    buf->register_consumer(unique_name);

    _base_dir = config.get<std::string>(unique_name, "base_dir");
    _file_name = config.get<std::string>(unique_name, "file_name");
    _file_ext = config.get<std::string>(unique_name, "file_ext");
    _num_frames_per_file = config.get_default<uint32_t>(unique_name, "num_frames_per_file", 1);
    _prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    _exit_after_n_files = config.get_default<uint32_t>(unique_name, "exit_after_n_files", 0);

    if (_exit_after_n_files > 0)
        waiting_for_max_frames++;
}

rawFileWrite::~rawFileWrite() {}

void rawFileWrite::main_thread() {

    int fd;
    uint32_t file_num = 0;
    uint32_t frame_id = 0;
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
        frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        // Check for NDArray on first frame (after producer may have set the descriptor)
        if (buf->get_frame_desc<kotekan::GenericNDArray>()) {
            FATAL_ERROR(
                "rawFileWrite does not support NDArray buffers. The NDArray frame descriptor "
                "is set dynamically and will not be written to the file.");
        }

        // Start timing the write time
        double st = current_time();

        if (!isFileOpen) {

            if (_prefix_hostname) {
                snprintf(full_path, full_path_len, "%s/%s_%s_%07u.%s", _base_dir.c_str(), hostname,
                         _file_name.c_str(), file_num, _file_ext.c_str());
            } else {
                snprintf(full_path, full_path_len, "%s/%s_%07u.%s", _base_dir.c_str(),
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
        std::shared_ptr<metadataObject> mc = buf->get_metadata(frame_id);
        if (mc)
            metadata_size = mc->get_serialized_size();
        // Write metadata size to disk, if there is no metadata in the frame, then
        // just save 0 to the first word.
        if (write(fd, (void*)&metadata_size, sizeof(metadata_size))
            != (int32_t)sizeof(metadata_size)) {
            ERROR("Failed to write metadata_size to disk for file {:s}", full_path);
            exit(-1);
        }
        if (mc) {
            char metabuf[metadata_size];
            mc->serialize(metabuf);
            if (write(fd, metabuf, metadata_size) != (int32_t)metadata_size) {
                ERROR("Failed to write metadata to disk for file {:s}", full_path);
                exit(-1);
            }
        }
        DEBUG("Wrote {:d} metadata bytes to {:s}.", metadata_size, full_path);

        // Write the contents of the buffer frame to disk.
        ssize_t bytes_writen = write(fd, frame, buf->frame_size);

        if ((size_t)bytes_writen != buf->frame_size) {
            ERROR("Failed to write buffer to disk for file {:s}", full_path);
            exit(-1);
        }
        DEBUG("Wrote {:d} data bytes to {:s}.", bytes_writen, full_path);

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

        buf->mark_frame_empty(unique_name, frame_id);

        // Check if we should exit after writing out a fixed number of files.
        // Useful for some tests and burst modes.  Uses the atomic "waiting_for_max_frames",
        // may be replaced by frames which can contain a "final" signal or some other
        // mechanism.
        if (_exit_after_n_files > 0 && file_num >= _exit_after_n_files) {

            // Unregister to allow the pipeline to continue, unless I'm the last
            // consumer on this buffer.
            buf->unregister_consumer(unique_name, true);

            if (--waiting_for_max_frames == 0)
                exit_kotekan(ReturnCode::CLEAN_EXIT);
            break;
        }

        frame_id = (frame_id + 1) % buf->num_frames;
    }

    DEBUG("Exiting");
}
