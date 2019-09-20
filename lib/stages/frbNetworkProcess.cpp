#include "frbNetworkProcess.hpp"

#include "Config.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "errors.h"
#include "fpga_header_functions.h"
#include "tx_utils.hpp"
#include "util.h"

#include <arpa/inet.h>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

using std::string;

// Update beam_offset parameter with:
// curl localhost:12048/frb/update_beam_offset -X POST -H 'Content-Type: application/json' -d
// '{"beam_offset":108}'

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(frbNetworkProcess);

frbNetworkProcess::frbNetworkProcess(Config& config_, const string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&frbNetworkProcess::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Apply config.
    udp_frb_packet_size = config.get_default<int>(unique_name, "udp_frb_packet_size", 4264);
    udp_frb_port_number = config.get_default<int>(unique_name, "udp_frb_port_number", 1313);
    number_of_nodes = config.get_default<int>(unique_name, "number_of_nodes", 256);
    number_of_subnets = config.get_default<int>(unique_name, "number_of_subnets", 4);
    packets_per_stream = config.get_default<int>(unique_name, "packets_per_stream", 8);
    beam_offset = config.get_default<int>(unique_name, "beam_offset", 0);
    time_interval = config.get_default<unsigned long>(unique_name, "time_interval", 125829120);
    column_mode = config.get_default<bool>(unique_name, "column_mode", false);
    samples_per_packet = config.get_default<int>(unique_name, "timesamples_per_frb_packet", 16);
}

frbNetworkProcess::~frbNetworkProcess() {
    restServer::instance().remove_json_callback("/frb/update_beam_offset");
}


void frbNetworkProcess::update_offset_callback(connectionInstance& conn, json& json_request) {
    // no need for a lock here, beam_offset copied into a local variable for use
    try {
        beam_offset = json_request["beam_offset"];
    } catch (...) {
        conn.send_error("Couldn't parse new beam_offset parameter.", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    INFO("Updating beam_offset to {:d}", beam_offset);
    conn.send_empty_reply(HTTP_RESPONSE::OK);
    config.update_value(unique_name, "beam_offset", beam_offset);
}

void frbNetworkProcess::main_thread() {
    DEBUG("number of subnets {:d}\n", number_of_subnets);

    int my_sequence_id = initialize_source_sockets();
    // check for errors initializing
    if (my_sequence_id < 0)
        return;

    int number_of_l1_links = initialize_destinations();
    INFO("number_of_l1_links: {:d}", number_of_l1_links);

    // rest server
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    string endpoint = "/frb/update_beam_offset";
    rest_server.register_post_callback(
        endpoint, std::bind(&frbNetworkProcess::update_offset_callback, this, _1, _2));

    // config.update_value(unique_name, "beam_offset", beam_offset);

    // declaring the timespec variables used mostly for the timing issues
    struct timespec t0, t1;
    t0.tv_sec = 0;
    t0.tv_nsec = 0; /*  nanoseconds */

    unsigned long time_interval =
        samples_per_packet * packets_per_stream * 384 * 2560; // time per buffer frame in ns
    // 384 is integration factor and 2560 fpga sampling time in ns

    long count = 0;


    int frame_id = 0;
    uint8_t* packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
    if (packet_buffer == nullptr)
        return;

    // waiting for atleast two frames for the buffer to fill up takes care of the random delay at
    // the start.

    clock_gettime(CLOCK_MONOTONIC, &t0);
    add_nsec(t0, 2 * time_interval); // time_interval is delay for each frame
    CLOCK_ABS_NANOSLEEP(CLOCK_MONOTONIC, t0);


    // time_interval value (125829120 ns) is divided into sections of 230 ns. Each node is assigned
    // a section according to the my_sequence_id. This will make sure that no two L0 nodes are
    // introducing packets to the network at the same time.

    clock_gettime(CLOCK_REALTIME, &t0);

    unsigned long abs_ns = t0.tv_sec * 1e9 + t0.tv_nsec;
    unsigned long reminder = (abs_ns % time_interval);
    unsigned long wait_ns =
        time_interval - reminder + my_sequence_id * 230; // analytically it must be 240.3173828125


    add_nsec(t0, wait_ns);

    CLOCK_ABS_NANOSLEEP(CLOCK_REALTIME, t0);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t* packet_buffer_uint64 = reinterpret_cast<uint64_t*>(packet_buffer);
    uint64_t initial_fpga_count = packet_buffer_uint64[1];
    uint64_t initial_nsec = t0.tv_sec * 1e9 + t0.tv_nsec;

    while (!stop_thread) {

        // reading the next frame and comparing the fpga clock with the monotonic clock.
        if (count != 0) {
            packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
            if (packet_buffer == NULL)
                break;
            packet_buffer_uint64 = reinterpret_cast<uint64_t*>(packet_buffer);
            clock_gettime(CLOCK_MONOTONIC, &t1);

            add_nsec(t0, time_interval);

            // discipline the monotonic clock with the fpga time stamps
            uint64_t offset = (t0.tv_sec * 1e9 + t0.tv_nsec - initial_nsec)
                              - (packet_buffer_uint64[1] - initial_fpga_count) * 2560;
            if (offset != 0)
                WARN("OFFSET in not zero ");
            add_nsec(t0, -1 * offset);
        }

        t1 = t0;

        int local_beam_offset = beam_offset;
        int beam_offset_upper_limit = 512;
        if (local_beam_offset > beam_offset_upper_limit) {
            WARN("Large beam_offset requested... capping at {:d}", beam_offset_upper_limit);
            local_beam_offset = beam_offset_upper_limit;
        }
        if (local_beam_offset < 0) {
            WARN("Negative beam_offset requested... setting to 0.");
            local_beam_offset = 0;
        }
        DEBUG("Beam offset: {:d}", local_beam_offset);

        for (int frame = 0; frame < packets_per_stream; frame++) {
            for (int stream = 0; stream < 256; stream++) {
                int e_stream = my_sequence_id
                               + stream; // making sure no two nodes send packets to same L1 node
                if (e_stream > 255)
                    e_stream -= 256;
                CLOCK_ABS_NANOSLEEP(CLOCK_MONOTONIC, t1);

                for (int link = 0; link < number_of_l1_links; link++) {
                    if (e_stream == local_beam_offset / 4 + link) {
                        auto dst = dst_sockets[link];
                        sendto(src_sockets[dst.sending_socket],
                               &packet_buffer[(e_stream * packets_per_stream + frame)
                                              * udp_frb_packet_size],
                               udp_frb_packet_size, 0, (struct sockaddr*)&dst.addr,
                               sizeof(dst.addr));
                    }
                }
                long wait_per_packet = (long)(50000);

                // 61521.25 is the theoretical seperation of packets in ns
                // I have used 58880 for convinence and also hope this will take care for
                // any clock glitches.

                add_nsec(t1, wait_per_packet);
            }
        }

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
        count++;
    }
    return;
}

int frbNetworkProcess::initialize_source_sockets() {
    int rack, node, nos, my_node_id;
    parse_chime_host_name(rack, node, nos, my_node_id);
    for (int i = 0; i < number_of_subnets; i++) {
        // construct an local address for each of the four VLANs 10.6.0.0/16..10.9.0.0/16
        std::string ip_addr = fmt::format("10.{:d}.{:d}.1{:d}", i + 6, nos + rack, node);
        DEBUG("{} ", ip_addr);

        // parse the local address and port into a `sockaddr` struct
        struct sockaddr_in addr;
        std::memset((char*)&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        inet_pton(AF_INET, ip_addr.c_str(), &addr.sin_addr);
        addr.sin_port = htons(udp_frb_port_number);

        // bind a sending UDP socket to the local address and port
        int sock_fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_fd < 0) {
            FATAL_ERROR("Network Thread: socket() failed: {:s} ", strerror(errno));
            return -1;
        }
        if (bind(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            FATAL_ERROR("port binding failed ");
            return -1;
        }

        // use a larger send buffer for the socket
        const int n = 256 * 1024 * 1024;
        if (setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, (void*)&n, sizeof(n)) < 0) {
            FATAL_ERROR("Network Thread: setsockopt() failed: %s ", strerror(errno));
            return -1;
        }

        src_sockets.push_back(sock_fd);
    }

    /* every node is introducing packets to the network. To achive load balancing a sequece_id is
       computed for each node this will make sure none of the catalyst switches are overloaded with
       traffic at a given point of time ofcourse this will be usefull when chrony is sufficuently
       synchronized across all nodes..
    */

    int my_sequence_id =
        (int)(my_node_id / 128) + 2 * ((my_node_id % 128) / 8) + 32 * (my_node_id % 8);
    return my_sequence_id;
}


int frbNetworkProcess::initialize_destinations() {
    // reading the L1 ip addresses from the config file
    const std::vector<std::string> link_ip =
        config.get<std::vector<std::string>>(unique_name, "L1_node_ips");
    for (size_t i = 0; i < link_ip.size(); i++) {
        sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        inet_pton(AF_INET, link_ip[i].c_str(), &addr.sin_addr);
        addr.sin_port = htons(udp_frb_port_number);
        int sending_socket = get_vlan_from_ip(link_ip[i].c_str()) - 6;
        dst_sockets.push_back({addr, sending_socket});
    }
    return link_ip.size();
}
