#include "fullPacketDump.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for ERROR, INFO
#include "restServer.hpp"      // for connectionInstance, restServer, HTTP_RESPONSE, HTTP_RESPO...

#include <atomic>     // for atomic_bool
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fcntl.h>    // for open, O_CREAT, O_WRONLY
#include <fmt.hpp>    // for format, fmt
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, function
#include <regex>      // for match_results<>::_Base_type
#include <stdio.h>    // for snprintf
#include <stdlib.h>   // for exit, free, malloc
#include <string.h>   // for memcpy
#include <unistd.h>   // for close, gethostname, sleep, write, ssize_t
#include <vector>     // for vector


#define MAX_NUM_PACKETS 100

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(fullPacketDump);

fullPacketDump::fullPacketDump(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&fullPacketDump::main_thread, this)) {

    link_id = config.get<int>(unique_name, "link_id");
    buf = get_buffer("network_in_buf");
    register_consumer(buf, unique_name.c_str());

    // Apply config.
    _packet_size = config.get<int>(unique_name, "udp_packet_size");
    _dump_to_disk = config.get<bool>(unique_name, "dump_to_disk");
    _file_base = config.get<std::string>(unique_name, "file_base");
    _data_set = config.get<std::string>(unique_name, "data_set");

    _packet_frame = (uint8_t*)malloc(_packet_size * MAX_NUM_PACKETS);

    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    endpoint = fmt::format(fmt("{:s}/packet_grab/{:d}"), unique_name, link_id);
    rest_server.register_post_callback(
        endpoint, std::bind(&fullPacketDump::packet_grab_callback, this, _1, _2));
}

fullPacketDump::~fullPacketDump() {
    restServer::instance().remove_json_callback(endpoint);
    free(_packet_frame);
}

void fullPacketDump::packet_grab_callback(connectionInstance& conn, nlohmann::json& json_request) {

    if (!got_packets) {
        conn.send_error("no packets captured yet.", HTTP_RESPONSE::REQUEST_FAILED);
        return;
    }

    int num_packets = 0;
    try {
        num_packets = json_request["num_packets"];
    } catch (...) {
        conn.send_error("could not parse/find num_packets parameter", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }

    if (num_packets > MAX_NUM_PACKETS || num_packets < 0) {
        conn.send_error("num_packets out of range", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    int len = num_packets * _packet_size;
    std::lock_guard<std::mutex> lock(_packet_frame_lock);
    conn.send_binary_reply((uint8_t*)_packet_frame, len);
}

void fullPacketDump::main_thread() {
    int frame_id = 0;

    int file_num = 0;
    char host_name[100];
    gethostname(host_name, 100);


    int first_time = 1;
    uint8_t* frame = nullptr;

    // Wait for, and drop full buffers
    while (!stop_thread) {

        // This call is blocking!
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        if (!_dump_to_disk) {
            std::lock_guard<std::mutex> lock(_packet_frame_lock);
            memcpy(_packet_frame, frame, _packet_size * MAX_NUM_PACKETS);
            if (!got_packets)
                got_packets = true;
        }

        if (_dump_to_disk) {

            if (first_time == 1) {
                sleep(5);
                first_time = 0;
            }

            const int file_name_len = 200;
            char file_name[file_name_len];

            snprintf(file_name, file_name_len, "%s/%s/%s_%d_%07d.pkt", _file_base.c_str(),
                     _data_set.c_str(), host_name, link_id, file_num);

            int fd = open(file_name, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: {:s}", file_name);
                exit(errno);
            }

            ssize_t bytes_writen = write(fd, frame, buf->frame_size);

            if (bytes_writen != buf->frame_size) {
                ERROR("Failed to write buffer to disk!!!  Abort, Panic, etc.");
                exit(-1);
            }

            if (close(fd) == -1) {
                ERROR("Cannot close file {:s}", file_name);
            }

            INFO("Data file write done for {:s}", file_name);
            file_num++;
        }

        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + 1) % buf->num_frames;
    }
    INFO("Closing full packet dump thread...");
}
