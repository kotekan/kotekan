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

#include <algorithm>  // for max
#include <atomic>     // for __atomic_base, atomic
#include <errno.h>    // for errno
#include <fcntl.h>    // for open, O_CREAT, O_WRONLY
#include <filesystem> // for directory_iterator (continue_numbering)
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
    // Opt-in: start numbering AFTER the highest <file_name>_NNNNNNN.<ext> already in base_dir
    // instead of at 0. The default restarts at 0 on every process start and open()s WITHOUT
    // O_TRUNC, so a restarted long-running archiver silently overwrites its own oldest files.
    // An archive meant to outlive the process must say so here.
    _continue_numbering = config.get_default<bool>(unique_name, "continue_numbering", false);
    // Opt-in escape from the NDArray refusal below. The refusal exists because the frame
    // descriptor is set dynamically and is not written to the file, so a reader cannot
    // recover the shape from the file alone. That is no hazard when the shape is fixed and
    // known out of band (a flat sample stream whose only "shape" is samples_per_data_set,
    // pinned in both the capture and the replay config). Default false: you must say you know.
    _allow_ndarray = config.get_default<bool>(unique_name, "allow_ndarray", false);
    // Opt-in: create base_dir and its parents now. The default leaves the directory to the
    // operator, but a writer that sits idle behind a gate for hours would otherwise take the
    // process down on a missing directory at the one moment the data matters.
    if (config.get_default<bool>(unique_name, "create_base_dir", false)) {
        std::error_code ec;
        std::filesystem::create_directories(_base_dir, ec);
        if (ec)
            FATAL_ERROR("rawFileWrite: cannot create base_dir {:s}: {:s}", _base_dir,
                        ec.message());
    }

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

    if (_continue_numbering) {
        // Files are "<prefix>_<7 digits>.<ext>"; resume one past the largest number seen.
        // Anything else in the directory (other prefixes, other runs) is ignored, so two
        // writers with different file_name values can share a base_dir.
        std::string prefix = _prefix_hostname
                                 ? fmt::format("{:s}_{:s}_", hostname, _file_name)
                                 : fmt::format("{:s}_", _file_name);
        std::string suffix = "." + _file_ext;
        std::error_code ec;
        for (auto& ent : std::filesystem::directory_iterator(_base_dir, ec)) {
            std::string n = ent.path().filename().string();
            if (n.size() != prefix.size() + 7 + suffix.size() || n.compare(0, prefix.size(), prefix)
                || n.compare(n.size() - suffix.size(), suffix.size(), suffix))
                continue;
            std::string digits = n.substr(prefix.size(), 7);
            if (digits.find_first_not_of("0123456789") != std::string::npos)
                continue;
            file_num = std::max(file_num, (uint32_t)std::stoul(digits) + 1);
        }
        if (ec)
            WARN("continue_numbering: cannot list {:s} ({:s}); starting at 0", _base_dir,
                 ec.message());
        else
            INFO("continue_numbering: resuming {:s}NNNNNNN{:s} at {:07d}", prefix, suffix,
                 file_num);
    }

    auto& write_time_metric =
        Metrics::instance().add_gauge("kotekan_rawfilewrite_write_time_seconds", unique_name);
    while (!stop_thread) {

        // This call is blocking.
        frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        // Check for NDArray on first frame (after producer may have set the descriptor)
        if (buf->get_frame_desc<kotekan::GenericNDArray>() && !_allow_ndarray) {
            FATAL_ERROR(
                "rawFileWrite does not support NDArray buffers. The NDArray frame descriptor "
                "is set dynamically and will not be written to the file. Set "
                "allow_ndarray: true to write the raw bytes anyway (the descriptor is then "
                "the READER's responsibility -- see the note on the option).");
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
