#include "BasebandWriter.hpp"

#include "StageFactory.hpp"
#include "visUtil.hpp"

#include <errno.h>    // for errno
#include <fcntl.h>    // for O_CREAT, O_WRONLY
#include <sys/stat.h> // for mkdir

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(BasebandWriter);

BasebandWriter::BasebandWriter(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&BasebandWriter::main_thread, this)),

    _root_path(config.get_default<std::string>(unique_name, "root_path", ".")),
    in_buf(get_buffer("in_buf")) {
    register_consumer(in_buf, unique_name.c_str());
}


void BasebandWriter::main_thread() {
    frameID frame_id(in_buf);

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Write the frame to its event+frequency destination file
        write_data(in_buf, frame_id);

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

        // TODO: Clean out any acquisitions that have been inactive for long enough
        // close_old_acqs();
    }
}

void BasebandWriter::write_data(Buffer* in_buf, int frame_id) {
    const auto frame = BasebandFrameView(in_buf, frame_id);
    const auto metadata = frame.metadata();

    INFO("Frame {} from {}/{}", metadata->fpga_seq, metadata->event_id, metadata->freq_id);

    const std::string event_directory_name =
        fmt::format("{:s}/baseband_raw_{:d}", _root_path, metadata->event_id);
    if (baseband_events.count(metadata->event_id) == 0) {
        mkdir(event_directory_name.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }

    const std::string file_name = fmt::format("{:s}/baseband_{:d}_{:d}", event_directory_name,
                                              metadata->event_id, metadata->freq_id);
    if (baseband_events[metadata->event_id].count(metadata->freq_id) == 0) {
        // NOTE: emplace the file instance or it will get closed by the destructor
        baseband_events[metadata->event_id].emplace(metadata->freq_id, file_name);
    }
    BasebandFileRaw& baseband_file = baseband_events[metadata->event_id].at(metadata->freq_id);
    ssize_t bytes_written = baseband_file.write_frame({in_buf, frame_id});

    if (bytes_written != in_buf->frame_size) {
        ERROR("Failed to write buffer to disk for file {:s}", file_name);
        exit(-1);
    }
}
