#include "pulsarNetworkProcess.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_empty, wait_for_full_frame, register_consumer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO, CHECK_MEM
#include "tx_utils.hpp"        // for add_nsec, get_vlan_from_ip, parse_chime_host_name, CLOCK_...
#include "vdif_functions.h"    // for VDIFHeader

#include <arpa/inet.h>  // for inet_pton
#include <atomic>       // for atomic_bool
#include <cstdio>       // for snprintf
#include <cstring>      // for memset
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <memory>       // for allocator_traits<>::value_type
#include <netinet/in.h> // for sockaddr_in, htons, IPPROTO_UDP
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for runtime_error
#include <stdint.h>     // for int64_t, uint8_t
#include <stdlib.h>     // for free, malloc
#include <string>       // for string, allocator
#include <sys/socket.h> // for AF_INET, bind, sendto, setsockopt, socket, SOCK_DGRAM
#include <sys/time.h>   // for CLOCK_MONOTONIC, CLOCK_REALTIME
#include <time.h>       // for timespec, clock_gettime
#include <vector>       // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using std::string;


REGISTER_KOTEKAN_STAGE(pulsarNetworkProcess);

pulsarNetworkProcess::pulsarNetworkProcess(Config& config_, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&pulsarNetworkProcess::main_thread, this)) {
    in_buf = get_buffer("pulsar_out_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Apply config.
    udp_pulsar_packet_size = config.get<int>(unique_name, "udp_pulsar_packet_size");
    udp_pulsar_port_number = config.get<int>(unique_name, "udp_pulsar_port_number");
    number_of_nodes = config.get<int>(unique_name, "number_of_nodes");
    number_of_subnets = config.get<int>(unique_name, "number_of_subnets");
    timesamples_per_pulsar_packet =
        config.get_default<int>(unique_name, "timesamples_per_pulsar_packet", 625);
    num_packet_per_stream = config.get_default<int>(unique_name, "num_packet_per_stream", 80);
    _num_beams = config.get<int>(unique_name, "num_beams");

    my_host_name = (char*)malloc(sizeof(char) * 100);
    CHECK_MEM(my_host_name);
}

pulsarNetworkProcess::~pulsarNetworkProcess() {
    free(my_host_name);
    for (int i = 0; i < number_of_subnets; i++)
        free(my_ip_address[i]);

    free(my_ip_address);
    free(socket_ids);
    free(myaddr);
    free(server_address);
    free(sock_fd);
}

void pulsarNetworkProcess::main_thread() {
    // parsing the host name

    int rack, node, nos, my_node_id;
    std::vector<std::string> link_ip =
        config.get<std::vector<std::string>>(unique_name, "pulsar_node_ips");
    int number_of_pulsar_links = link_ip.size();
    INFO("number_of_pulsar_links: {:d}", number_of_pulsar_links);

    // Allocating buffers
    sock_fd = (int*)malloc(sizeof(int) * number_of_subnets);
    server_address = (sockaddr_in*)malloc(sizeof(sockaddr_in) * number_of_pulsar_links);
    myaddr = (sockaddr_in*)malloc(sizeof(sockaddr_in) * number_of_pulsar_links);

    socket_ids = (int*)malloc(sizeof(int) * number_of_pulsar_links);

    my_ip_address = (char**)malloc(sizeof(char*) * number_of_subnets);
    for (int i = 0; i < number_of_subnets; i++)
        my_ip_address[i] = (char*)malloc(sizeof(char) * 100);
    // std::stringstream temp_ip[number_of_subnets];
    INFO("number of subnets {:d}\n", number_of_subnets);

    // parsing the host name

    parse_chime_host_name(rack, node, nos, my_node_id);
    for (int i = 0; i < number_of_subnets; i++) {
        if (std::snprintf(my_ip_address[i], 100, "10.%d.%d.1%d", i + 15, nos + rack, node) > 100) {
            FATAL_ERROR("buffer spill over ");
            return;
        }
        INFO("{:s} ", my_ip_address[i]);
    }

    int frame_id = 0;
    uint8_t* packet_buffer = nullptr;

    for (int i = 0; i < number_of_subnets; i++) {
        sock_fd[i] = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

        if (sock_fd[i] < 0) {
            FATAL_ERROR("network thread: socket() failed: ");
            return;
        }
    }


    for (int i = 0; i < number_of_subnets; i++) {
        std::memset((char*)&myaddr[i], 0, sizeof(myaddr[i]));

        myaddr[i].sin_family = AF_INET;
        inet_pton(AF_INET, my_ip_address[i], &myaddr[i].sin_addr);

        myaddr[i].sin_port = htons(udp_pulsar_port_number);

        // Binding port to the socket
        if (bind(sock_fd[i], (struct sockaddr*)&myaddr[i], sizeof(myaddr[i])) < 0) {
            FATAL_ERROR("port binding failed ");
            return;
        }
    }

    for (int i = 0; i < number_of_pulsar_links; i++) {
        memset(&server_address[i], 0, sizeof(server_address[i]));
        server_address[i].sin_family = AF_INET;
        inet_pton(AF_INET, link_ip[i].c_str(), &server_address[i].sin_addr);
        server_address[i].sin_port = htons(udp_pulsar_port_number);
        socket_ids[i] = get_vlan_from_ip(link_ip[i].c_str()) - 15;
    }

    int n = 256 * 1024 * 1024;
    for (int i = 0; i < number_of_subnets; i++) {
        if (setsockopt(sock_fd[i], SOL_SOCKET, SO_SNDBUF, (void*)&n, sizeof(n)) < 0) {
            FATAL_ERROR("network thread: setsockopt() failed ");
            return;
        }
    }

    struct timespec t0, t1;
    t0.tv_sec = 0;
    t0.tv_nsec = 0; /*  nanoseconds */

    unsigned long time_interval =
        num_packet_per_stream * timesamples_per_pulsar_packet * 2560; // time per buffer frame in ns
    // 2560 is fpga sampling time in ns

    int my_sequence_id =
        (int)(my_node_id / 128) + 2 * ((my_node_id % 128) / 8) + 32 * (my_node_id % 8);

    packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
    if (packet_buffer == nullptr)
        return;
    mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
    frame_id = (frame_id + 1) % in_buf->num_frames;

    clock_gettime(CLOCK_REALTIME, &t0);

    unsigned long abs_ns = t0.tv_sec * 1e9 + t0.tv_nsec;
    unsigned long reminder = (abs_ns % time_interval);
    unsigned long wait_ns =
        time_interval - reminder + my_sequence_id * 600; // analytically it must be 781.25


    add_nsec(t0, wait_ns);

    CLOCK_ABS_NANOSLEEP(CLOCK_REALTIME, t0);

    clock_gettime(CLOCK_MONOTONIC, &t0);

    // added to take care of the missed frames
    VDIFHeader* header = reinterpret_cast<VDIFHeader*>(packet_buffer);
    int64_t vdif_last_seconds = header->seconds;
    int64_t vdif_last_frame = header->data_frame;

    while (!stop_thread) {
        packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
        if (packet_buffer == nullptr)
            break;

        header = reinterpret_cast<VDIFHeader*>(packet_buffer);
        time_interval = 2560
                        * (390625 * (header->seconds - vdif_last_seconds)
                           + 625 * (header->data_frame - vdif_last_frame));

        add_nsec(t0, time_interval);
        t1.tv_sec = t0.tv_sec;
        t1.tv_nsec = t0.tv_nsec;

        vdif_last_seconds = header->seconds;
        vdif_last_frame = header->data_frame;

        for (int frame = 0; frame < 80; frame++) {
            for (int beam = 0; beam < _num_beams; beam++) {
                int e_beam = my_sequence_id + beam;
                e_beam = e_beam % _num_beams;
                CLOCK_ABS_NANOSLEEP(CLOCK_MONOTONIC, t1);
                if (e_beam < number_of_pulsar_links) {
                    sendto(sock_fd[socket_ids[e_beam]],
                           &packet_buffer[(e_beam)*80 * udp_pulsar_packet_size
                                          + frame * udp_pulsar_packet_size],
                           udp_pulsar_packet_size, 0, (struct sockaddr*)&server_address[e_beam],
                           sizeof(server_address[e_beam]));
                }

                long wait_per_packet = (long)(153600);

                // 61521.25 is the theoritical seperation of packets in ns
                // I have used 61440 for convinence and also hope this will take care for
                // any clock glitches.
                add_nsec(t1, wait_per_packet);
            }
        }

        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
    return;
}
