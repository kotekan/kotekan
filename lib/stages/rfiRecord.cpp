#include "rfiRecord.hpp"

#include "chimeMetadata.h"
#include "errors.h"
#include "gpsTime.h"
#include "util.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(rfiRecord);

rfiRecord::rfiRecord(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiRecord::main_thread, this)) {
    // Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    // Register stage as consumer
    register_consumer(rfi_buf, unique_name.c_str());

    // General config parameters
    _num_freq = config.get_default<uint32_t>(unique_name, "num_total_freq", 1024);
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI config parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _rfi_combined = config.get_default<bool>(unique_name, "rfi_combined", true);
    // Stage-specific parameters
    _total_links = config.get_default<uint32_t>(unique_name, "total_links", 1);
    _write_to = config.get<std::string>(unique_name, "write_to");
    _write_to_disk = config.get_default<bool>(unique_name, "write_to_disk", false);
    _frames_per_file = config.get_default<uint32_t>(unique_name, "frames_per_file", 1024);

    // Initialize rest server endpoint
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    endpoint = unique_name + "/rfi_record";
    rest_server.register_post_callback(endpoint,
                                       std::bind(&rfiRecord::rest_callback, this, _1, _2));
}

rfiRecord::~rfiRecord() {
    restServer::instance().remove_json_callback(endpoint);
}

void rfiRecord::rest_callback(connectionInstance& conn, json& json_request) {
    // Notify request was received
    WARN("RFI Record Callback Received... Changing Parameters")
    // Lock callback mutex
    rest_callback_mutex.lock();
    // Update parameters
    _write_to = json_request["write_to"].get<string>();
    WARN("write_to {:s}", _write_to)
    _write_to_disk = json_request["write_to_disk"].get<bool>();
    WARN("write_to_disk: {:d}", _write_to_disk)
    // This will trigger main process to update directories
    file_num = 0;
    //    file_num = 2048*(int)((file_num + 2048)/2048);
    // Update Config Values
    config.update_value(unique_name, "write_to", _write_to);
    config.update_value(unique_name, "write_to_disk", _write_to_disk);
    // Send reply indicating success
    conn.send_empty_reply(HTTP_RESPONSE::OK);
    // Unlock mutex
    rest_callback_mutex.unlock();
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

        if (_write_to_disk) {

            fpga_seq_num = get_fpga_seq_num(rfi_buf, frame_id);
            stream_id = get_stream_id_t(rfi_buf, frame_id);

            // Only write a new file every _frames_per_file frames
            if (fpga_seq_num % (_samples_per_data_set * _frames_per_file) == 0) {

                started = true;

                char data_time[64];
                struct timespec gps_time = get_gps_time(rfi_buf, frame_id);
                struct tm timeinfo;
                if (gmtime_r(&gps_time.tv_sec, &timeinfo) == NULL) {
                    ERROR("Cannot gerate time info");
                }
                strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%S", &timeinfo);
                snprintf(file_name, sizeof(file_name), "%s/%sN%09ld_%d.rfi", _write_to.c_str(),
                         data_time, gps_time.tv_nsec, bin_number_chime(&stream_id));

                // Close the old file if it's open
                if (fd != -1) {
                    INFO("Closing last file");
                    close(fd);
                }

                // Open the new file
                INFO("Opening file: %s", file_name);
                fd = open(file_name, O_WRONLY | O_APPEND | O_CREAT, 0666);
                if (fd < 0) {
                    ERROR("Cannot open file {:s}, error {:d} ({:s}})", file_name, errno,
                          strerror(errno));
                }
                // Write seq number into the start of the file
                ssize_t bytes_writen = write(fd, &fpga_seq_num, sizeof(int64_t));
                if (bytes_writen != sizeof(int64_t)) {
                    ERROR("Failed to write seq_num to disk");
                }
            }

            if (started) {
                ssize_t bytes_writen = write(fd, frame, rfi_buf->frame_size);
                if (bytes_writen != rfi_buf->frame_size) {
                    ERROR("Failed to write buffer to disk");
                }
            }
        }
        // Unlock callback mutex
        rest_callback_mutex.unlock();
        // Mark Frame Empty
        mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
        // Move forward one frame/link/file
        frame_id = (frame_id + 1) % rfi_buf->num_frames;
    }
}
