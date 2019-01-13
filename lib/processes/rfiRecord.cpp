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

REGISTER_KOTEKAN_PROCESS(rfiRecord);

rfiRecord::rfiRecord(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiRecord::main_thread, this)) {
    // Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    // Register process as consumer
    register_consumer(rfi_buf, unique_name.c_str());

    // General config parameters
    _num_freq = config.get_default<uint32_t>(unique_name, "num_total_freq", 1024);
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI config parameters
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _rfi_combined = config.get_default<bool>(unique_name, "rfi_combined", true);
    // Process specific parameters
    _total_links = config.get_default<uint32_t>(unique_name, "total_links", 1);
    _write_to = config.get<std::string>(unique_name, "write_to");
    _write_to_disk = config.get_default<bool>(unique_name, "write_to_disk", false);

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
    WARN("write_to %s", _write_to.c_str())
    _write_to_disk = json_request["write_to_disk"].get<bool>();
    WARN("write_to_disk: %d", _write_to_disk)
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

void rfiRecord::save_meta_data(uint16_t streamID, int64_t firstSeqNum, timeval tv, timespec ts) {
    // Create Directories
    char data_time[50];
    time_t rawtime;
    struct tm* timeinfo;
    // Get current time and format it
    time(&rawtime);
    timeinfo = gmtime(&rawtime);

    // TODO: switch to using fmt
    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(time_dir, sizeof(time_dir), "%s_rfi", data_time);
    // Call to utils to that actually makes the directories
    make_rfi_dirs((int)streamID, _write_to.c_str(), time_dir);
    // Create Info File
    char info_file_name[200];
    snprintf(info_file_name, sizeof(info_file_name), "%s/%d/%s/info.txt", _write_to.c_str(),
             streamID, time_dir);
    FILE* info_file = fopen(info_file_name, "w");
    if (!info_file) {
        ERROR("Error creating info file: %s\n", info_file_name);
    }
    // Populate Info file with information
    fprintf(info_file, "utcTime=%s\n", data_time);
    fprintf(info_file, "streamID=%d\n", streamID);
    fprintf(info_file, "firstSeqNum=%lld\n", (long long)firstSeqNum);
    fprintf(info_file, "first_packet_gps_tv_sec=%ld\n", ts.tv_sec);
    fprintf(info_file, "first_packet_gps_tv_nsec=%09ld\n", ts.tv_nsec);
    fprintf(info_file, "first_packet_ntp_tv_sec=%ld\n", tv.tv_sec);
#ifndef MAC_OSX
    fprintf(info_file, "first_packet_ntp_tv_usec=%06ld\n", tv.tv_usec);
#else
    fprintf(info_file, "first_packet_ntp_tv_usec=%06d\n", tv.tv_usec);
#endif
    fprintf(info_file, "num_elements=%d\n", _num_elements);
    fprintf(info_file, "num_total_freq=%d\n", _num_freq);
    fprintf(info_file, "num_local_freq=%d\n", _num_local_freq);
    fprintf(info_file, "samples_per_data_set=%d\n", _samples_per_data_set);
    fprintf(info_file, "sk_step=%d\n", _sk_step);
    fprintf(info_file, "rfi_combined=%d\n", _rfi_combined);
    fprintf(info_file, "total_links=%d\n", _total_links);
    // Close Info file
    fclose(info_file);
    INFO("Created meta data file: %s\n", info_file_name);
}
void rfiRecord::main_thread() {
    // Initialize variables
    uint32_t frame_id = 0;
    uint8_t* frame = NULL;
    uint32_t link_id = 0;
    int64_t fpga_seq_num;
    int fd = -1;
    file_num = 0;
    // Endless Loop
    while (!stop_thread) {
        // Get Frame
        frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
        if (frame == NULL)
            break;
        // Lock mutex
        rest_callback_mutex.lock();
        // Reset Timer
        double start_time = e_time();
        fpga_seq_num = get_fpga_seq_num(rfi_buf, frame_id);
        // Only write if user specifilly asks (Just for caution)
        if (_write_to_disk) {
            // For each link
            if (file_num < _total_links) {
                INFO("ATTEMPTING TO CREATE METADATA");
                // Make Necessary Directories using timecode and create info file with metadata
                save_meta_data(get_stream_id(rfi_buf, frame_id), fpga_seq_num,
                               get_first_packet_recv_time(rfi_buf, frame_id),
                               get_gps_time(rfi_buf, frame_id));
                //                save_meta_data((uint16_t)link_id, get_fpga_seq_num(rfi_buf,
                //                frame_id));
            }
            // Initialize file name
            char file_name[100];
            // Figure out which file (adjust file name every 1024 buffers)
            snprintf(file_name, sizeof(file_name), "%s/%d/%s/%07d.rfi", _write_to.c_str(),
                     get_stream_id(rfi_buf, frame_id),
                     //                     link_id,
                     time_dir, file_num / 1024);
            // Open that file
            fd = open(file_name, O_WRONLY | O_APPEND | O_CREAT, 0666);
            if (fd < 0) {
                ERROR("Cannot open file %s", file_name);
            } else {
                // Write buffer to that file
                ssize_t bytes_writen = write(fd, &fpga_seq_num, sizeof(int64_t));
                if (bytes_writen != sizeof(int64_t)) {
                    ERROR("Failed to write seq_num to disk");
                }
                bytes_writen = write(fd, frame, rfi_buf->frame_size);
                if (bytes_writen != rfi_buf->frame_size) {
                    ERROR("Failed to write buffer to disk");
                }
                // Close that file
                if (close(fd) < 0) {
                    ERROR("Cannot close file %s", file_name);
                } else {
                    INFO("Frame ID %d Succesfully Recorded link %d out of %d links in %fms",
                         frame_id, link_id + 1, _total_links, (e_time() - start_time) * 1000);
                }
            }
        }
        // Unlock callback mutex
        rest_callback_mutex.unlock();
        // Mark Frame Empty
        mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
        // Move forward one frame/link/file
        frame_id = (frame_id + 1) % rfi_buf->num_frames;
        link_id = (link_id + 1) % _total_links;
        file_num++;
    }
}
