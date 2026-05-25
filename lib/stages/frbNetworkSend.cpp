#include "frbNetworkSend.hpp"

#include "Config.hpp"            // for Config
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chordMetadata.hpp"     // for get_chord_metadata
#include "frb_functions.h"       // for FRBHeader
#include "kotekanLogging.hpp"    // for DEBUG, INFO, WARN, FATAL_ERROR, ERROR
#include "network_functions.hpp" // for receive_ping, send_ping
#include "restServer.hpp"        // for restServer, HTTP_RESPONSE, connectionInstance
#include "tx_utils.hpp"          // for add_nsec, CLOCK_ABS_NANOSLEEP, get_vlan_from_ip, parse_...

#include "fmt.hpp" // for compile_string_to_view, format, format_string

#include <algorithm>    // for max, max_element
#include <arpa/inet.h>  // for inet_pton, htons, inet_ntop
#include <assert.h>     // for assert
#include <cstring>      // for strerror, memset, size_t
#include <errno.h>      // for errno, EINTR
#include <map>          // for map, operator!=, _Rb_tree_iterator
#include <mutex>        // for mutex, unique_lock
#include <pthread.h>    // for pthread_setaffinity_np
#include <queue>        // for priority_queue
#include <random>       // for random_device, uniform_int_distribution, mt19937
#include <sched.h>      // for cpu_set_t, CPU_SET, CPU_ZERO
#include <string>       // for basic_string, allocator, string
#include <sys/select.h> // for FD_SET, FD_ZERO, select, FD_ISSET, fd_set
#include <sys/socket.h> // for AF_INET, bind, socket, sendto, setsockopt, SOCK_DGRAM
#include <sys/time.h>   // for CLOCK_MONOTONIC, CLOCK_REALTIME, timeval
#include <thread>       // for thread
#include <time.h>       // for timespec, clock_gettime
#include <unistd.h>     // for close
#include <utility>      // for get, move, pair


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

REGISTER_KOTEKAN_STAGE(frbNetworkSend);

frbNetworkSend::frbNetworkSend(Config& config_, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&frbNetworkSend::main_thread, this)),
    _ping_interval{
        std::chrono::seconds(config_.get_default<uint32_t>(unique_name, "ping_interval", 360))},
    _quick_ping_interval{
        std::chrono::seconds(config_.get_default<uint32_t>(unique_name, "quick_ping_interval", 5))},
    _ping_dead_threshold{std::chrono::seconds(
        config_.get_default<uint32_t>(unique_name, "ping_dead_threshold", 30))},
    tel(Telescope::instance()) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    offsetscale_buf = get_buffer("offsetscale_buf");
    offsetscale_buf->register_consumer(unique_name);

    // Apply config.
    udp_frb_packet_size = config.get_default<int>(unique_name, "udp_frb_packet_size", 4272);
    udp_frb_port_number = config.get_default<int>(unique_name, "udp_frb_port_number", 1313);
    number_of_nodes = config.get_default<int>(unique_name, "number_of_nodes", 256);
    number_of_subnets = config.get_default<int>(unique_name, "number_of_subnets", 4);
    packets_per_stream = config.get_default<int>(unique_name, "packets_per_stream", 8);
    beam_offset = config.get_default<int>(unique_name, "beam_offset", 0);
    column_mode = config.get_default<bool>(unique_name, "column_mode", false);
    samples_per_packet = config.get_default<int>(unique_name, "timesamples_per_frb_packet", 16);
    num_frequencies = config.get<int>(unique_name, "num_frequencies");

    assert(samples_per_packet == _timesamples_per_frb_packet);
    assert(packets_per_stream == 8); // as in CHIME production, more here, but will chunk up like this

    assert(num_frequencies % _nfreq_coarse == 0);
    if (_ping_dead_threshold != std::chrono::seconds::zero()) {
        INFO("Pinging every {} / {}", _quick_ping_interval, _ping_interval);
    } else {
        INFO("L1 ping check is disabled");
    }
}

frbNetworkSend::~frbNetworkSend() {
    restServer::instance().remove_json_callback("/frb/update_beam_offset");

    for (auto src : src_sockets) {
        close(src.socket_fd);
    }

    // close pinging sockets
    for (auto fd : ping_src_fd) {
        close(fd);
    }
}


void frbNetworkSend::update_offset_callback(connectionInstance& conn,
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

void frbNetworkSend::main_thread() {
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
        endpoint, std::bind(&frbNetworkSend::update_offset_callback, this, _1, _2));

    // config.update_value(unique_name, "beam_offset", beam_offset);

    int frame_id = 0;
    uint8_t* beams_buffer = (uint8_t*)in_buf->wait_for_full_frame(unique_name, frame_id);
    if (beams_buffer == nullptr)
        return;
    float16_t* offsetscale_buffer = (float16_t*)offsetscale_buf->wait_for_full_frame(unique_name, frame_id);
    if (offsetscale_buffer == nullptr)
        return;

    auto metadata = get_chord_metadata(in_buf, frame_id);
    auto in_coarse_freq = metadata->get_coarse_freq();
    const auto beams_frame_desc = in_buf->get_ndarray_frame_desc();
    const auto offsetscale_frame_desc = offsetscale_buf->get_ndarray_frame_desc();

    // check that we have cast to the correct type
    assert(kotekan::GetDataType_v<std::remove_reference_t<decltype(*beams_buffer)>> == beams_frame_desc->get_value_datatype());
    assert(kotekan::GetDataType_v<std::remove_reference_t<decltype(*offsetscale_buffer)>> == offsetscale_frame_desc->get_value_datatype());

    // 384 is integration factor and 2560 fpga sampling time in ns
    const uint32_t fpga_ns = tel.seq_length_nsec();
    const int time_downsampling_fpga = metadata->get_time_downsampling_fpga();
    assert(time_downsampling_fpga == 384); // for now
    const unsigned samples_per_frame =
        samples_per_packet * packets_per_stream
        * time_downsampling_fpga; // number of FPGA samples in each frame
    const unsigned long time_interval = samples_per_frame * fpga_ns; // time per buffer frame in ns

    long count = 0;

    // check expected frame layout
    // TODO: this could easily handle any value for R4, Fbar64, Ttilde16_lo16
    auto const expected_beams_frame_desc = kotekan::GenericNDArray::create(
        kotekan::DataType::uint8, "I3",
        {1, _total_nbeams / _nbeams, num_frequencies / _nfreq_coarse,
         16 /*TODO: get from option? */ , _nbeams, _nfreq_coarse,
         _factor_upchan_out, _timesamples_per_frb_packet},
        {"Ttilde256",     "R4",        "Fbar64", "Ttilde16_lo16", "Rlo4",      "Fbar16_lo4", "Fbarlo16",      "Ttildelo16"}, nullptr);
    assert(*beams_frame_desc == *expected_beams_frame_desc);

    auto const expected_offsetscale_frame_desc = kotekan::GenericNDArray::create(
        kotekan::DataType::float16, "I3_offsetscale",
        {1, _total_nbeams / _nbeams, num_frequencies / _nfreq_coarse,
         16 /*TODO: get from option? */ , _nbeams, _nfreq_coarse, 2},
        {"Ttilde256", "R4", "Fbar64", "Ttilde16_lo16", "Rlo4", "Fbar16_lo4", "offset/scale"}, nullptr);
    assert(*offsetscale_frame_desc == *expected_offsetscale_frame_desc);

    // waiting for at least two frames for the buffer to fill up takes care of the random delay at
    // the start.

    // declaring the timespec variables used mostly for the timing issues
    struct timespec t0 {
        .tv_sec = 0, .tv_nsec = 0
    }, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    add_nsec(t0, 2 * time_interval); // time_interval is delay for each frame
    CLOCK_ABS_NANOSLEEP(CLOCK_MONOTONIC, t0);


    // time_interval value (125829120 ns) is divided into sections of 230 ns. Each node is assigned
    // a section according to the my_sequence_id. This will make sure that no two L0 nodes are
    // introducing packets to the network at the same time.

    clock_gettime(CLOCK_REALTIME, &t0);

    unsigned long abs_ns = t0.tv_sec * 1'000'000'000UL + t0.tv_nsec;
    unsigned long reminder = (abs_ns % time_interval);
    // TODO: understand and remove the magic number 230
    unsigned long wait_ns =
        time_interval - reminder + my_sequence_id * 230; // analytically it must be 240.3173828125


    add_nsec(t0, wait_ns);

    CLOCK_ABS_NANOSLEEP(CLOCK_REALTIME, t0);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t initial_nsec = t0.tv_sec * 1'000'000'000UL + t0.tv_nsec;

    const uint64_t initial_fpga_count = metadata->get_fpga_seq_num();
    uint64_t last_fpga_count = initial_fpga_count;

    while (!stop_thread) {

        // reading the next frame and comparing the fpga clock with the monotonic clock.
        if (count != 0) {
            beams_buffer = (uint8_t*)in_buf->wait_for_full_frame(unique_name, frame_id);
            if (beams_buffer == nullptr)
                break;
            offsetscale_buffer =
                (float16_t*)offsetscale_buf->wait_for_full_frame(unique_name, frame_id);
            if (offsetscale_buffer == nullptr)
                break;

            metadata = get_chord_metadata(in_buf, frame_id);
            in_coarse_freq = metadata->get_coarse_freq();

            clock_gettime(CLOCK_MONOTONIC, &t1);

            add_nsec(t0, time_interval);

            // discipline the monotonic clock with the fpga time stamps
            const uint64_t fpga_samples_skipped =
                metadata->get_fpga_seq_num() - last_fpga_count - samples_per_frame;
            if (fpga_samples_skipped) {
                const auto frames_skipped = fpga_samples_skipped / samples_per_frame;
                INFO("Adjust pacing clock for {} skipped frames", frames_skipped);
                uint64_t nanos_skipped = frames_skipped * time_interval;
                add_nsec(t0, nanos_skipped);
            }
            uint64_t offset = (t0.tv_sec * 1e9 + t0.tv_nsec - initial_nsec)
                              - (metadata->get_fpga_seq_num() - initial_fpga_count) * fpga_ns;
            if (offset != 0)
                WARN("OFFSET in not zero ");
            add_nsec(t0, -1 * offset);

            last_fpga_count = metadata->get_fpga_seq_num();
        }

        t1 = t0;

        int local_beam_offset = beam_offset;
        const int beam_offset_upper_limit = 512;
        if (local_beam_offset > beam_offset_upper_limit) {
            WARN("Large beam_offset requested... capping at {:d}", beam_offset_upper_limit);
            local_beam_offset = beam_offset_upper_limit;
        }
        if (local_beam_offset < 0) {
            WARN("Negative beam_offset requested... setting to 0.");
            local_beam_offset = 0;
        }
        DEBUG("Beam offset: {:d}", local_beam_offset);

        // r4 is e_stream (or link) below
        for (int fbar64 = 0; fbar64 < num_frequencies / _nfreq_coarse; ++fbar64) // TODO: think about swapping this with the ttilde16_low16_by_packets_per_stream loop to keep all frequencies closer together in the network stream
        for (int ttilde16_low16_by_packets_per_stream = 0; ttilde16_low16_by_packets_per_stream < 16/packets_per_stream; ++ttilde16_low16_by_packets_per_stream)
        for (int frame = 0; frame < packets_per_stream; frame++) {
              const int nstreams = _total_nbeams / _nbeams;
              assert(nstreams == 256); 
              for (int stream = 0; stream < nstreams; stream++) {
                  int e_stream =
                      (my_sequence_id + stream)
                      % (_total_nbeams
                         / _nbeams); // making sure no two nodes send packets to same L1 node
                  assert(_total_nbeams / _nbeams == nstreams);
                  CLOCK_ABS_NANOSLEEP(CLOCK_MONOTONIC, t1);

                  for (int link = 0; link < number_of_l1_links; link++) {
                      if (e_stream
                          == local_beam_offset / 4 + link) { // RH: not sure what the `/4` means.
                          DestIpSocket& dst = stream_dest.at(link);
                          if (dst.active
                              && (_ping_dead_threshold == std::chrono::seconds::zero()
                                  || dst.live)) {

                              const int r4 = e_stream;
                              const int ttilde16_low16 = ttilde16_low16_by_packets_per_stream * packets_per_stream + frame;
                              const int frb_packet_num =
                                r4 * num_frequencies / _nfreq_coarse * 16 +
                                fbar64 * 16 +
                                ttilde16_low16;

                              // TODO: this is mostly constant and could be mostly
                              // moved outside of this inner loop
                              static_assert(sizeof(FRBHeader) == 32);
                              const size_t variable_size_part =
                                  sizeof(uint16_t) * _nbeams +       // beam_ids
                                  sizeof(uint16_t) * _nfreq_coarse + // coarse_freq_ids
                                  // TODO: see how the quantizer generates these
                                  sizeof(float) * _nbeams * _nfreq_coarse + // scale
                                  sizeof(float) * _nbeams * _nfreq_coarse;  // offset
                              std::vector<char> header_buf(sizeof(FRBHeader)
                                                           + variable_size_part);

                              FRBHeader& header(
                                  *reinterpret_cast<FRBHeader*>(&header_buf.front()));
                              header = FRBHeader{
                                  .protocol_version = 2,
                                  .data_nbytes = _nbeams * _nfreq_coarse * _factor_upchan_out
                                                 * _timesamples_per_frb_packet,
                                  .fpga_counts_per_sample =
                                      static_cast<uint16_t>(time_downsampling_fpga),
                                  .fpga0_ns = static_cast<uint64_t>(
                                      tel.to_time_ns(0)), // a constant over the run
                                  .fpga_count =
                                      metadata->get_fpga_seq_num()
                                      + frame * _timesamples_per_frb_packet
                                            * static_cast<uint64_t>(time_downsampling_fpga),
                                  .nbeams = _nbeams,
                                  .nfreq_coarse = _nfreq_coarse, // 4
                                  .nupfreq = _factor_upchan_out,
                                  .ntsamp = _timesamples_per_frb_packet};

                              uint16_t* beam_ids = reinterpret_cast<uint16_t*>(&header + 1);
                              // beam id convention 0->255, 1000->1255, 2000->2255, 3000->3255
                              for(int b = 0; b < _nbeams; ++b) {
                                const int beam_id = e_stream * _nbeams + b;
                                beam_ids[b] =  (beam_id) % 256 + (int((beam_id) / 256) * 1000);
                              }

                              uint16_t* coarse_freq_ids =
                                  reinterpret_cast<uint16_t*>(beam_ids + _nbeams);
                              // for unchannelized data coarse_freq holds the
                              // source frequency for each fine channel, so
                              // repeats _factor_upchan_out times
                              for(int f = 0; f < _nfreq_coarse; ++f) {
                                  coarse_freq_ids[f] = in_coarse_freq.at((fbar64 * _nfreq_coarse + f) * _factor_upchan_out);
                                  for(int ff = 0; ff < _factor_upchan_out; ++ff) {
                                      assert(coarse_freq_ids[f] == in_coarse_freq.at((fbar64 * _nfreq_coarse + f) * _factor_upchan_out + ff));
                                  }
                              }

                              float* scale =
                                  reinterpret_cast<float*>(coarse_freq_ids + _nfreq_coarse);
                              float* offset =
                                  reinterpret_cast<float*>(scale + _nbeams * _nfreq_coarse);
                              for (int b = 0; b < _nbeams; ++b) {
                                  for (int f = 0; f < _nfreq_coarse; ++f) {
                                      const ptrdiff_t idx_out = b * _nfreq_coarse + f;
                                      const ptrdiff_t idx_in = frb_packet_num * _nbeams * _nfreq_coarse + idx_out;
                                      assert(idx_out < _nbeams * _nfreq_coarse);
                                      assert(idx_in < num_frequencies * _total_nbeams);

                                      offset[idx_out] =
                                          static_cast<float>(offsetscale_buffer[2 * idx_in + 0]);
                                      scale[idx_out] =
                                          static_cast<float>(offsetscale_buffer[2 * idx_in + 1]);
                                  }
                              }

                              assert(reinterpret_cast<char*>(offset + _nbeams * _nfreq_coarse)
                                     == &*header_buf.cend());

                              struct iovec msg_iov[2] = {
                                  {.iov_base = header_buf.data(), .iov_len = header_buf.size()},
                                  {.iov_base =
                                       &beams_buffer[frb_packet_num * header.data_nbytes],
                                   .iov_len = static_cast<size_t>(header.data_nbytes)},
                              };
                              struct msghdr hdr = {.msg_name = (void*)&dst.addr,
                                                   .msg_namelen = sizeof(dst.addr),
                                                   .msg_iov = msg_iov,
                                                   .msg_iovlen =
                                                       sizeof(msg_iov) / sizeof(msg_iov[0]),
                                                   .msg_control = nullptr,
                                                   .msg_controllen = 0,
                                                   .msg_flags = 0};
                              ssize_t expected_send_len = 0;
                              for(size_t i = 0; i < hdr.msg_iovlen; ++i)
                                  expected_send_len += hdr.msg_iov[i].iov_len;

                              const ssize_t send_len =
                                  sendmsg(src_sockets[dst.sending_socket].socket_fd, &hdr, 0);
                              if (send_len != expected_send_len) {
                                  char ip_addr[64] = "<unknown>";
                                  ERROR("Error sending to {:s}: {:d}",
                                        inet_ntop(AF_INET, hdr.msg_name, ip_addr, sizeof(ip_addr)),
                                        send_len);
                              }
                          }
                      }
                  }
                  // TODO: this may need adjustment since we are sending more data
                  long wait_per_packet = (long)(50000);

                  // 61521.25 is the theoretical seperation of packets in ns
                  // I have used 58880 for convinence and also hope this will take care for
                  // any clock glitches.

                  add_nsec(t1, wait_per_packet);
            }
        }

        in_buf->mark_frame_empty(unique_name, frame_id);
        offsetscale_buf->mark_frame_empty(unique_name, frame_id);
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

int frbNetworkSend::initialize_source_sockets() {
    int rack, node, nos, my_node_id;
    if (number_of_subnets > 0)
        parse_chime_host_name(rack, node, nos, my_node_id);
    else
        rack = node = nos = my_node_id = 0;
    for (int i = 0; i < (number_of_subnets > 0 ? number_of_subnets : 1); i++) {
        // construct an local address for each of the four VLANs 10.6.0.0/16..10.9.0.0/16
        std::string ip_addr = number_of_subnets > 0
                                  ? fmt::format("10.{:d}.{:d}.1{:d}", i + 6, nos + rack, node)
                                  : "127.0.0.1";
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
        if (number_of_subnets > 0) {
            if (bind(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
                FATAL_ERROR("port binding failed ");
                return -1;
            }
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


int frbNetworkSend::initialize_destinations() {
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
                const int vlan = get_vlan_from_ip(link_ip[i].c_str());
                // in CHIME FRB L1 is on VLANs 10.6.0.0/16..10.9.0.0/16
                const int sending_socket = number_of_subnets > 0 ? vlan - 6 : 0;
                assert(sending_socket >= 0 && sending_socket < (ptrdiff_t)src_sockets.size()
                       && "unexpected vlan");
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

void frbNetworkSend::initialize_pinging_sockets() {
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


void frbNetworkSend::ping_destinations() {
    if (ping_src_fd.empty())
        return;
    // quick destination lookup by next scheduled check time
    using RefDestIpSocketTime = std::reference_wrapper<DestIpSocketTime>;

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
            if (send_ping(ping_src_fd.at(lru_dest.dst->sending_socket), lru_dest.dst->addr,
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

void frbNetworkSend::receive_ping_responses() {
    if (ping_src_fd.empty())
        return;

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
