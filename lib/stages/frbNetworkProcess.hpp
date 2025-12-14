/**
 * @file frbNetworkProcess.hpp
 * @brief Network transmission stage for FRB obs
 *  - frbNetworkProcess : public kotekan::Stage
 */

#ifndef FRBNETWORKPROCESS_HPP
#define FRBNETWORKPROCESS_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "restServer.hpp"      // for connectionInstance

#include "json.hpp" // for json

#include <atomic>             // for atomic_bool
#include <bits/chrono.h>      // for operator==, operator>, seconds, steady_clock, time_point
#include <condition_variable> // for condition_variable
#include <functional>         // for reference_wrapper
#include <map>                // for map
#include <netinet/in.h>       // for sockaddr_in, in_addr
#include <stdint.h>           // for uint32_t, uint16_t
#include <string>             // for string, basic_string
#include <vector>             // for vector

/**
 * @class frbNetworkProcess
 * @brief frbNetworkProcess Network transmission stage for FRB obs
 *
 *
 * This is an Kotekan stage that read packetized data from frbPostProcess and transmits 1024 beams
 * to 256 links of frb backend. frbNetworkProcess distributes the out going traffic to four VLANS
 * (10.6 10.7 10.8 10.9) on single 1 Gig port. The frb total data rate is ~0.55 gbps. The node IP
 * address is derived by parsing the hostname.
 *
 * @par REST Endpoints
 * @endpoint POST /frb/update_beam_offset {"beam_offset": <int>} to adjust the beam index offset.
 *
 * @par Buffers
 * @buffer in_buf The kotkean buffer to hold the packets to be transmitted to L1 nodes
 * 	@buffer_format Array of unsigned char.
 * 	@buffer_metadata none
 *
 *
 * @conf   in_buf               String. Input packet buffer.
 * @conf   udp_frb_packet_size  Int (default 4272). Packet size including header.
 * @conf   udp_frb_port_number  Int (default 1313). UDP port number for FRB streams.
 * @conf   number_of_nodes      Int (default 256). Number of L0/L1 destinations.
 * @conf   number_of_subnets    Int (default 4). Number of VLANS/subnets to shard traffic across.
 * @conf   packets_per_stream   Int (default 8). Packets per stream within each frame.
 * @conf   beam_offset          Int (default 0). Offset applied to beam IDs going to L1.
 * @conf   time_interval        Unsigned long (default 125829120). Time per buffer in ns.
 * @conf   column_mode          Bool (default false). Send beams in a single CHIME cylinder.
 * @conf   timesamples_per_frb_packet Int (default 16). Samples per packet.
 * @conf   cpu_affinity         Array<Int>. CPU cores to pin ping threads (if pinging enabled).
 * @conf   L1_node_ips          Array<String>. List of destination IPs (one per stream).
 * @conf   ping_interval        UInt32 (default 360). Seconds between pings when live.
 * @conf   quick_ping_interval  UInt32 (default 5). Seconds between pings when a node stops
 * responding.
 * @conf   ping_dead_threshold  UInt32 (default 30). Seconds of quick checking before declaring
 * dead; 0 disables ping checks.
 *
 * @par Example
 * @code
 * frbNetworkProcess:
 *   in_buf: frb_packets
 *   udp_frb_packet_size: 4272
 *   udp_frb_port_number: 1313
 *   number_of_nodes: 256
 *   number_of_subnets: 4
 *   packets_per_stream: 8
 *   beam_offset: 0
 *   time_interval: 125829120
 *   column_mode: false
 *   timesamples_per_frb_packet: 16
 *   cpu_affinity: [0,1]
 *   L1_node_ips: ["10.6.0.1", "10.6.0.2"]
 *   ping_interval: 360
 *   quick_ping_interval: 5
 *   ping_dead_threshold: 30
 * @endcode
 *
 * @todo   Resolve the issue of NTP clock vs Monotonic clock.
 *
 * @author Arun Naidu, Davor Cubranic
 *
 */

struct SrcAddrSocket {
    const sockaddr_in addr;
    const int socket_fd;
};


/**
 * @brief Convenience struct used to hold all relevant information about an FRB L1 destination
 */
struct DestIpSocket {
    /// Regular constructor used with data from the config file
    DestIpSocket(std::string host, sockaddr_in addr, int s, bool active = true);

    /// Move constructor is necessary for inserting into standard containers
    DestIpSocket(DestIpSocket&& other);

    //@{
    /// host address as a std::string and a `sockaddr` structure
    const std::string host;
    const sockaddr_in addr;
    //@}

    /// index of the entry in @p src_sockets used to communicate with the destination
    const int sending_socket;

    /// flag to indicate if the destination is a "dummy" placeholder
    const bool active;

    /// flag to indicate if the host has been responding to pings
    std::atomic_bool live;
};

/**
 * @brief internal data type for keeping track of host checks and replies
 */
struct DestIpSocketTime {
    DestIpSocket* dst;
    std::chrono::steady_clock::time_point last_responded;
    std::chrono::steady_clock::time_point next_check;
    std::chrono::steady_clock::time_point last_checked = last_responded;
    uint16_t ping_seq = 0;
    friend bool operator<(const DestIpSocketTime& l, const DestIpSocketTime& r) {
        if (l.next_check == r.next_check) {
            // break check time ties by host address
            return l.dst->addr.sin_addr.s_addr > r.dst->addr.sin_addr.s_addr;
        } else
            return l.next_check > r.next_check;
    }
};

class frbNetworkProcess : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    frbNetworkProcess(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);

    /// Destructor , cleaning local allocations
    virtual ~frbNetworkProcess();

    /// Callback to update the beam offset
    void update_offset_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

    /// Callback to change destination active status
    void set_destination_active_callback(kotekan::connectionInstance& conn,
                                         nlohmann::json& json_request);

    /// main thread
    void main_thread() override;

private:
    /// pointer to Input FRB buffer
    Buffer* in_buf;

    /// frb packet size
    int udp_frb_packet_size;

    /// port number
    int udp_frb_port_number;

    /// number of L0 nodes
    int number_of_nodes;

    /// number of VLANS
    int number_of_subnets;

    /// number of packets to each L1 nodes
    int packets_per_stream;

    /// beam offset for 8-node frb system
    int beam_offset;

    // time per buffer frame in ns
    unsigned long time_interval;

    // samples per packet
    int samples_per_packet;

    // Beam kotekan::Configuration Mode
    bool column_mode;

    /// Interval between checks of a node's liveliness
    const std::chrono::seconds _ping_interval;

    /// Accelerated interval between checks of a node's liveliness, used when a live node stops
    /// responding
    const std::chrono::seconds _quick_ping_interval;

    /// Duration at which a node is declared dead if it hasn't responded to pings
    const std::chrono::seconds _ping_dead_threshold;

    /// array of sending socket descriptors
    std::vector<SrcAddrSocket> src_sockets;

    /// destination addresses and associated sending sockets, indexed by IP @c s_addr
    std::map<uint32_t, DestIpSocket> dest_sockets;

    /// stream destinations (references to @p dest_sockets, because a single destination can be used
    /// for multiple streams)
    std::vector<std::reference_wrapper<DestIpSocket>> stream_dest;

    /// raw sockets used as sources for outgoing pings
    std::vector<int> ping_src_fd;

    // quick destination lookup by IP address
    std::map<uint32_t, DestIpSocketTime> dest_by_ip;

    /// initialize sockets used to send data to FRB nodes
    int initialize_source_sockets();

    /// initialize destination addresses and determine the sending socket to use
    int initialize_destinations();

    /// initialize raw sockets used for pinging
    void initialize_pinging_sockets();

    /// background thread that periodically pings destination hosts and updates their @c live status
    void ping_destinations();

    /// background thread that listens for ping replies from destination hosts
    void receive_ping_responses();

    /// used by @p ping_destinations for periodic sleep interruptible by the @p main_thread on
    /// Kotekan stop
    std::condition_variable ping_cv;
};

#endif
