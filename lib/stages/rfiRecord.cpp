#include "rfiRecord.hpp"

#include "chimeMetadata.h"
#include "configUpdater.hpp"
#include "errors.h"
#include "gpsTime.h"

#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rfiRecord);

rfiRecord::rfiRecord(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiRecord::main_thread, this)) {
    // Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    // Register stage as consumer
    register_consumer(rfi_buf, unique_name.c_str());

    // General config parameters
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _frames_per_file = config.get_default<uint32_t>(unique_name, "frames_per_file", 1024);

    // Initialize rest server endpoint
    using namespace std::placeholders;
    kotekan::configUpdater::instance().subscribe(this,
                                                 std::bind(&rfiRecord::config_callback, this, _1));
}

rfiRecord::~rfiRecord() {}

bool rfiRecord::config_callback(json& json) {

    string output_dir;
    bool write_to_disk;
    try {
        output_dir = json["output_dir"].get<string>();
        write_to_disk = json["write_to_disk"].get<bool>();
    } catch (std::exception& e) {
        ERROR("Failure parsing message. Error: {:s}, Request JSON: {:s}", e.what(), json.dump());
        return false;
    }
    rest_callback_mutex.lock();
    _output_dir = output_dir;
    _write_to_disk = write_to_disk;
    rest_callback_mutex.unlock();

    INFO("Updated RFI record parameters to: output_dir: {:s}, write_to_disk: {}", _output_dir,
         _write_to_disk);

    return true;
}

void rfiRecord::main_thread() {
    // Initialize variables
    uint32_t frame_id = 0;
    uint8_t* frame = nullptr;
    int64_t fpga_seq_num;
    stream_id_t stream_id;
    int fd = -1;
    // File name
    char file_name[100];
    bool started = false;
    // Endless Loop
    while (!stop_thread) {
        // Get Frame
        frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
        if (frame == NULL)
            break;
        // Lock mutex
        rest_callback_mutex.lock();

        if (_write_to_disk || started) {

            fpga_seq_num = get_fpga_seq_num(rfi_buf, frame_id);
            stream_id = get_stream_id_t(rfi_buf, frame_id);

            // Only write a new file every _frames_per_file frames
            if (fpga_seq_num % (_samples_per_data_set * _frames_per_file) == 0) {

                started = true;

                // Close the old file if it's open
                if (fd != -1) {
                    INFO("Closing last file");
                    close(fd);

                    // If we set `_write_to_disk` to false, but are currently writing
                    // i.e. `started` is true, then we want to finish writing the current
                    // file.   So only we only stop writing on the close state.
                    if (!_write_to_disk) {
                        started = false;
                        fd = -1;
                        rest_callback_mutex.unlock();
                        // Since both started and write to disk are false, the next time
                        // _write_to_disk is true, we start at a new edge with a new name.
                        continue;
                    }
                }

                char data_time[64];
                struct timespec gps_time = get_gps_time(rfi_buf, frame_id);
                struct tm timeinfo;
                if (gmtime_r(&gps_time.tv_sec, &timeinfo) == NULL) {
                    ERROR("Cannot gerate time info");
                }
                strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%S", &timeinfo);
                snprintf(file_name, sizeof(file_name), "%s/%sN%09ld_%d.rfi", _output_dir.c_str(),
                         data_time, gps_time.tv_nsec, bin_number_chime(&stream_id));

                rest_callback_mutex.unlock();

                // Open the new file
                INFO("Opening file: {:s}", file_name);
                fd = open(file_name, O_WRONLY | O_APPEND | O_CREAT, 0666);
                if (fd < 0) {
                    ERROR("Cannot open file {:s}, error {:d} ({:s})", file_name, errno,
                          strerror(errno));
                }
                // Write seq number into the start of the file
                ssize_t bytes_writen = write(fd, &fpga_seq_num, sizeof(int64_t));
                if (bytes_writen != sizeof(int64_t)) {
                    ERROR("Failed to write seq_num to disk");
                }
            } else {
                rest_callback_mutex.unlock();
            }

            if (started) {
                ssize_t bytes_writen = write(fd, frame, rfi_buf->frame_size);
                if (bytes_writen != rfi_buf->frame_size) {
                    ERROR("Failed to write buffer to disk");
                }
            }
        } else {
            rest_callback_mutex.unlock();
        }

        // Mark Frame Empty
        mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
        // Move forward one frame/link/file
        frame_id = (frame_id + 1) % rfi_buf->num_frames;
    }
}
