#include "frbNetworkProcess.hpp"

#include "Config.hpp"       // for Config
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.h"              // for wait_for_full_frame, mark_frame_empty, register_consumer
#include "bufferContainer.hpp"   // for bufferContainer
#include "frb_functions.h"       // for FRBHeader
#include "kotekanLogging.hpp"    // for DEBUG, INFO, WARN, FATAL_ERROR, ERROR
#include "network_functions.hpp" // for receive_ping, send_ping
#include "restServer.hpp"        // for restServer, connectionInstance, HTTP_RESPONSE, HTTP_RES...
#include "tx_utils.hpp"          // for add_nsec, CLOCK_ABS_NANOSLEEP, get_vlan_from_ip, parse_...

#include "fmt.hpp" // for format

#include <algorithm>    // for max, max_element, copy
#include <arpa/inet.h>  // for inet_pton, inet_ntop
#include <assert.h>     // for assert
#include <chrono>       // for operator+, operator-, seconds, steady_clock::time_point
#include <cstring>      // for strerror, memset, size_t
#include <errno.h>      // for errno, EINTR
#include <exception>    // for exception
#include <map>          // for map, map<>::mapped_type
#include <memory>       // for allocator_traits<>::value_type
#include <mutex>        // for mutex, unique_lock
#include <pthread.h>    // for pthread_setaffinity_np
#include <queue>        // for priority_queue
#include <random>       // for mt19937, random_device, uniform_int_distribution
#include <ratio>        // for ratio
#include <regex>        // for match_results<>::_Base_type
#include <sched.h>      // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>    // for runtime_error
#include <string>       // for string
#include <sys/select.h> // for select, FD_SET, FD_ZERO, FD_ISSET, fd_set
#include <sys/socket.h> // for AF_INET, bind, socket, sendto, setsockopt, SOCK_DGRAM
#include <sys/time.h>   // for CLOCK_MONOTONIC, CLOCK_REALTIME, timeval
#include <thread>       // for thread
#include <time.h>       // for clock_gettime, timespec
#include <type_traits>  // for __success_type<>::type
#include <unistd.h>     // for close
#include <utility>      // for move, get


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

frbNetworkProcess::frbNetworkProcess(Config& config_, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&frbNetworkProcess::main_thread, this)),
    _ping_interval{
        std::chrono::seconds(config_.get_default<uint32_t>(unique_name, "ping_interval", 360))},
    _quick_ping_interval{
        std::chrono::seconds(config_.get_default<uint32_t>(unique_name, "quick_ping_interval", 5))},
    _ping_dead_threshold{std::chrono::seconds(
        config_.get_default<uint32_t>(unique_name, "ping_dead_threshold", 30))} {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Apply config.
    udp_frb_packet_size = config.get_default<int>(unique_name, "udp_frb_packet_size", 4272);
    udp_frb_port_number = config.get_default<int>(unique_name, "udp_frb_port_number", 1313);
    number_of_nodes = config.get_default<int>(unique_name, "number_of_nodes", 256);
    number_of_subnets = config.get_default<int>(unique_name, "number_of_subnets", 4);
    packets_per_stream = config.get_default<int>(unique_name, "packets_per_stream", 8);
    beam_offset = config.get_default<int>(unique_name, "beam_offset", 0);
    time_interval = config.get_default<unsigned long>(unique_name, "time_interval", 125829120);
    column_mode = config.get_default<bool>(unique_name, "column_mode", false);
    samples_per_packet = config.get_default<int>(unique_name, "timesamples_per_frb_packet", 16);
    if (_ping_dead_threshold != std::chrono::seconds::zero()) {
        INFO("Pinging every {} / {}", _quick_ping_interval, _ping_interval);
    } else {
        INFO("L1 ping check is disabled");
    }
}

frbNetworkProcess::~frbNetworkProcess() {
    restServer::instance().remove_json_callback("/frb/update_beam_offset");

    for (auto src : src_sockets) {
        close(src.socket_fd);
    }

    // close pinging sockets
    for (auto fd : ping_src_fd) {
        close(fd);
    }
}


void frbNetworkProcess::update_offset_callback(connectionInstance& conn,
                                               nlohmann::json& json_request) {
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

    std::thread send_ping_thread;
    std::thread receive_ping_thread;
    if (_ping_dead_threshold != std::chrono::seconds::zero()) {
        initialize_pinging_sockets();

        send_ping_thread = std::thread([&] { this->ping_destinations(); });
        receive_ping_thread = std::thread([&] { this->receive_ping_responses(); });
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
            CPU_SET(i, &cpuset);

        pthread_setaffinity_np(send_ping_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        pthread_setaffinity_np(receive_ping_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // rest server
    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    std::string endpoint = "/frb/update_beam_offset";
    rest_server.register_post_callback(
        endpoint, std::bind(&frbNetworkProcess::update_offset_callback, this, _1, _2));

    // config.update_value(unique_name, "beam_offset", beam_offset);

    // declaring the timespec variables used mostly for the timing issues
    struct timespec t0, t1;
    t0.tv_sec = 0;
    t0.tv_nsec = 0; /*  nanoseconds */

    // 384 is integration factor and 2560 fpga sampling time in ns
    const uint32_t fpga_ns = Telescope::instance().seq_length_nsec();
    const unsigned samples_per_frame =
        samples_per_packet * packets_per_stream * 384; // number of FPGA samples in each frame
    unsigned long time_interval = samples_per_frame * fpga_ns; // time per buffer frame in ns

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
    uint64_t initial_nsec = t0.tv_sec * 1e9 + t0.tv_nsec;

    FRBHeader* header = reinterpret_cast<FRBHeader*>(packet_buffer);
    const uint64_t initial_fpga_count = header->fpga_count;
    uint64_t last_fpga_count = initial_fpga_count;

    while (!stop_thread) {

        // reading the next frame and comparing the fpga clock with the monotonic clock.
        if (count != 0) {
            packet_buffer = wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);
            if (packet_buffer == nullptr)
                break;

            clock_gettime(CLOCK_MONOTONIC, &t1);

            add_nsec(t0, time_interval);

            // discipline the monotonic clock with the fpga time stamps
            header = reinterpret_cast<FRBHeader*>(packet_buffer);
            const uint64_t fpga_samples_skipped =
                header->fpga_count - last_fpga_count - samples_per_frame;
            if (fpga_samples_skipped) {
                const auto frames_skipped = fpga_samples_skipped / samples_per_frame;
                INFO("Adjust pacing clock for {} skipped frames", frames_skipped);
                uint64_t nanos_skipped = frames_skipped * time_interval;
                add_nsec(t0, nanos_skipped);
            }
            uint64_t offset = (t0.tv_sec * 1e9 + t0.tv_nsec - initial_nsec)
                              - (header->fpga_count - initial_fpga_count) * fpga_ns;
            if (offset != 0)
                WARN("OFFSET in not zero ");
            add_nsec(t0, -1 * offset);

            last_fpga_count = header->fpga_count;
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
                        DestIpSocket& dst = stream_dest[link];
                        if (dst.active
                            && (_ping_dead_threshold == std::chrono::seconds::zero() || dst.live)) {
                            sendto(src_sockets[dst.sending_socket].socket_fd,
                                   &packet_buffer[(e_stream * packets_per_stream + frame)
                                                  * udp_frb_packet_size],
                                   udp_frb_packet_size, 0, (struct sockaddr*)&dst.addr,
                                   sizeof(dst.addr));
                        }
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
    ping_cv.notify_all();

    if (send_ping_thread.joinable()) {
        send_ping_thread.join();
    }
    if (receive_ping_thread.joinable()) {
        receive_ping_thread.join();
    }
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

        src_sockets.push_back({addr, sock_fd});
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
        if (!link_ip[i].empty()) {
            addr.sin_family = AF_INET;
            inet_pton(AF_INET, link_ip[i].c_str(), &addr.sin_addr);
            if (!dest_sockets.count(addr.sin_addr.s_addr)) {
                // new destination, initialize the entry in `dest_sockets`
                addr.sin_port = htons(udp_frb_port_number);
                int sending_socket = get_vlan_from_ip(link_ip[i].c_str()) - 6;
                dest_sockets.insert(
                    {addr.sin_addr.s_addr, DestIpSocket{link_ip[i], addr, sending_socket}});
            }
        } else {
            if (!dest_sockets.count(0)) {
                dest_sockets.insert({0, DestIpSocket{link_ip[i], addr, -1, false}});
            }
        }
        stream_dest.push_back(std::ref(dest_sockets.at(addr.sin_addr.s_addr)));
    }
    return link_ip.size();
}

void frbNetworkProcess::initialize_pinging_sockets() {
    bool err = false;

    for (auto& src : src_sockets) {
        int s = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
        if (s < 0) {
            ERROR("Cannot create source socket for FRB host pings (requires root). Stopping the "
                  "pings.");
            err = true;
        } else if (bind(s, (struct sockaddr*)&src.addr, sizeof(src.addr)) < 0) {
            ERROR("Cannot bind source socket for FRB host pings (requires root). Stopping the "
                  "pings.");
            err = true;
        } else {
            ping_src_fd.push_back(s);
        }
    }

    if (err) {
        // close any pinging sockets that were already created
        for (auto fd : ping_src_fd) {
            close(fd);
        }

        // clear the list so the main thread knows this failed
        ping_src_fd.clear();

        return;
    }

    // Random number generators to jitter the first ping time between hosts (3-30 s)
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(3000, 30000);

    auto now = std::chrono::steady_clock::now();
    for (auto& ipaddr_dst : dest_sockets) {
        // don't even check the inactive destinations
        DestIpSocket& dst = std::get<1>(ipaddr_dst);
        if (!dst.active) {
            continue;
        }
        uint32_t ipaddr = std::get<0>(ipaddr_dst);
        // jitter the initial check by a random amount 3-30 s
        auto next_check = now + std::chrono::milliseconds(dis(gen));
        DEBUG("Check host {} in {:%M:%S}", dst.host, next_check - now);
        dest_by_ip[ipaddr] = {&dst, now, next_check};
    }
}


using RefDestIpSocketTime = std::reference_wrapper<DestIpSocketTime>;

void frbNetworkProcess::ping_destinations() {
    assert(!ping_src_fd.empty());
    // quick destination lookup by next scheduled check time
    std::priority_queue<RefDestIpSocketTime> dest_by_time;
    for (auto& ipaddr_dst : dest_by_ip) {
        DestIpSocketTime& dest_ping_info = std::get<1>(ipaddr_dst);
        dest_by_time.push(dest_ping_info);
    }
    const auto startup_time = dest_by_time.top().get().last_responded;
    bool startup_phase = true;

    // it's silly to have a mutex local to a thread, but we need it for the condition variable used
    // by the main_thread to interrupt this thread's sleep to notify of Kotekan stopping
    std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);
    uint16_t ping_seq = 0;
    while (!stop_thread) {
        // sleep until the next host is due
        DestIpSocketTime& lru_dest = dest_by_time.top();
        auto now = std::chrono::steady_clock::now();
        while (!stop_thread && lru_dest.next_check > now) {
            DEBUG("Sleep for {:%M:%S} before checking the next host {}", lru_dest.next_check - now,
                  lru_dest.dst->host);

            // continue sleeping if received a spurious wakeup, i.e., neither the stage was stopped
            // nor timeout occurred
            if (ping_cv.wait_for(lock, lru_dest.next_check - now) == std::cv_status::timeout) {
                break;
            }
            now = std::chrono::steady_clock::now();
        }
        const auto time_since_last_live = now - lru_dest.last_responded;

        /*
         * The startup phase lasts for the first `_ping_interval`. During this
         * time, we send out pings on a quick_check schedule even though the
         * host is not live yet.
         */
        if (startup_phase && (now - startup_time > _ping_interval)) {
            startup_phase = false;
        }
        // special handling for hosts during the startup phase until they go live
        const bool host_startup = startup_phase && !lru_dest.dst->live;

        DEBUG("Checking: {} ({}, last responded {:%M:%S} ago)", lru_dest.dst->host,
              (lru_dest.dst->live ? "LIVE" : "DEAD"), time_since_last_live);

        if (host_startup || time_since_last_live >= (_ping_interval - _quick_ping_interval)) {
            // time to ping again
            if (send_ping(ping_src_fd[lru_dest.dst->sending_socket], lru_dest.dst->addr,
                          ping_seq)) {
                DEBUG("Pinged {}", lru_dest.dst->host);
                lru_dest.last_checked = now;
                lru_dest.ping_seq = ping_seq;
            } else {
                DEBUG("Pinging {} failed, will retry soon", lru_dest.dst->host);
            }
            ping_seq++;
        } else {
            DEBUG("Don't need to ping {} yet", lru_dest.dst->host);
        }

        // Schedule the next check for the node
        dest_by_time.pop();
        if (host_startup) {
            DEBUG("Schedule a follow-up check for starting-up host {} in {}", lru_dest.dst->host,
                  _quick_ping_interval);
            lru_dest.next_check = now + _quick_ping_interval;
        } else if (lru_dest.dst->live) {
            if (time_since_last_live >= _ping_interval + _ping_dead_threshold) {
                INFO("Too long since last ping response ({:%M:%S}), mark host {} dead.",
                     time_since_last_live, lru_dest.dst->host);
                lru_dest.dst->live = false;
                lru_dest.next_check = now + _ping_interval;
            } else if (time_since_last_live >= _ping_interval - _ping_dead_threshold) {
                DEBUG("Live host {} has not responded recently, schedule a backup check in {}",
                      lru_dest.dst->host, _quick_ping_interval);
                lru_dest.next_check = now + _quick_ping_interval;
            } else {
                DEBUG("Schedule a regular check for live host {} in {}", lru_dest.dst->host,
                      _ping_interval);
                lru_dest.next_check = lru_dest.last_checked + _ping_interval;
            }
        } else {
            DEBUG("Schedule a regular check for dead host {} in {}", lru_dest.dst->host,
                  _ping_interval);
            lru_dest.next_check = lru_dest.last_checked + _ping_interval;
        }
        // NOTE: could add a small random delay to next_check
        dest_by_time.push(lru_dest);
    }
}

void frbNetworkProcess::receive_ping_responses() {
    assert(!ping_src_fd.empty());
    const int max_ping_src_fd = *std::max_element(ping_src_fd.begin(), ping_src_fd.end());

    while (!stop_thread) {
        // initialize the listening set of sockets for `select`
        fd_set rfds;
        FD_ZERO(&rfds);
        for (int s : ping_src_fd) {
            FD_SET(s, &rfds);
        }

        /*
         * Minimally-blocking check for ping responses received on any of the `ping_src_fd` sockets:
         * wait for a response for no more than 0.7 s, which should be enough to receive replies
         * from nodes on the cluster, and then process all sockets that `select` marked as ready to
         * be read.
         */
        struct timeval tv = {0, 700'000}; // Don't wait more than 0.7s for a socket to be ready
        while (int rc = select(max_ping_src_fd + 1, &rfds, nullptr, nullptr, &tv) != 0) {
            if (stop_thread)
                break;
            if (rc < 0) {
                if (errno == EINTR) {
                    DEBUG("Select interrupted, try again");
                    continue;
                } else {
                    // TODO: stop we crash out at this point?
                    WARN("Ping listening error: {}.", errno);
                    break;
                }
            }
            for (int s : ping_src_fd) {
                if (FD_ISSET(s, &rfds)) {
                    sockaddr_in from; // out-param for `recv_from(2)`
                    int reply_ping_seq = receive_ping(s, from);
                    if (reply_ping_seq >= 0) {
                        if (dest_by_ip.count(from.sin_addr.s_addr)) {
                            DestIpSocketTime& src = dest_by_ip[from.sin_addr.s_addr];
                            if (reply_ping_seq == src.ping_seq) {
                                auto now = std::chrono::steady_clock::now();
                                DEBUG("Received ping response from: {}. Update `last_responded`.",
                                      src.dst->host);
                                src.last_responded = now;
                                if (!src.dst->live) {
                                    INFO("Host {} is responding to pings again, mark live.",
                                         src.dst->host);
                                    src.dst->live = true;
                                }
                            } else {
                                DEBUG(
                                    "Ignore ping reply with old sequence number ({}) from host {}",
                                    reply_ping_seq, src.dst->host);
                            }
                        } else {
#ifdef DEBUGGING
                            char src_addr_str[INET_ADDRSTRLEN + 1]; // Used for decoding host IP in
                                                                    // logging statements
#endif                                                              // DEBUGGING
                            DEBUG("Received ping response from unknown host: {}. Ignored.",
                                  inet_ntop(AF_INET, &from.sin_addr, src_addr_str,
                                            INET_ADDRSTRLEN + 1));
                        }
                    }
                }
            }
            tv = {0, 300'000}; // reduce the wait for additional replies
            FD_ZERO(&rfds);
            for (int s : ping_src_fd) {
                FD_SET(s, &rfds);
            }
        }
    }
}

DestIpSocket::DestIpSocket(std::string host, sockaddr_in addr, int s, bool active) :
    host(std::move(host)),
    addr(std::move(addr)),
    sending_socket(s),
    active(active),
    live(false) {}

DestIpSocket::DestIpSocket(DestIpSocket&& other) :
    host(std::move(other.host)),
    addr(std::move(other.addr)),
    sending_socket(other.sending_socket),
    active(other.active),
    live(other.live.load()) {}
