/**
 * @file frbNetworkProcess.hpp
 * @brief Network transmission stage for FRB obs
 *  - frbNetworkProcess : public kotekan::Stage
 */

#ifndef FRBNETWORKPROCESS_HPP
#define FRBNETWORKPROCESS_HPP

#include "Stage.hpp"
#include "buffer.h"
#include "restServer.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

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
 * @endpoint /frb/update_gains/``gpu_id`` Any contact here triggers a re-parse of the gains file.
 * @endpoint /frb/update_destination Set the active status of param ``host`` to value of param
 * ``active``. Data is sent only to active hosts.
 *
 * @par Buffers
 * @buffer in_buf The kotkean buffer to hold the packets to be transmitted to L1 nodes
 * 	@buffer_format Array of unsigned char.
 * 	@buffer_metadata none
 *
 *
 * @conf   udp_frb_packet_size  Int (default 4264). packet size including header
 * @conf   udp_frb_port_number  Int (default 1313). udp Port number for frb streams
 * @conf   number_of_nodes      Int (default 256). Number of L0 nodes
 * @conf   number_of_subnets    Int (default 4). Number of subnets or VLANS used for transmission of
 * FRB data
 * @conf   packets_per_stream   Int (default 8). Number of subnets or VLANS used for transmission of
 * FRB data
 * @conf   L1_node_ips          Array of Strings. List of IPs to send to. (?)
 * @conf   beam_offset          Int (default 0). Offset the beam_id going to L1 Process
 * @conf   time_interval        Unsigned long (default 125829120). Time per buffer in ns.
 * @conf   column_mode          bool (default false) Send beams in a single CHIME cylinder.
 * @conf   live_check_frequency Unsigned long (default 30s) Time in ms between sending a ping to
 * check if a destination host is live.
 * @conf   node_dead_interval   Unsigned long (default 2*live_check_frequency) Time in ms after
 * which the host is marked dead if it hasn't sent a ping reply. check if a destination host is
 * live.
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
    /// host address as a string and a `sockaddr` structure
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

class frbNetworkProcess : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    frbNetworkProcess(kotekan::Config& config, const string& unique_name,
                      kotekan::bufferContainer& buffer_container);

    /// Destructor , cleaning local allocations
    virtual ~frbNetworkProcess();

    /// Callback to update the beam offset
    void update_offset_callback(kotekan::connectionInstance& conn, json& json_request);

    /// Callback to change destination active status
    void set_destination_active_callback(kotekan::connectionInstance& conn, json& json_request);

    /// main thread
    void main_thread() override;

private:
    /// pointer to Input FRB buffer
    struct Buffer* in_buf;

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

    /// Minimal interval between checks of a node's liveliness
    const std::chrono::seconds _min_ping_interval;

    /// Maximal interval between checks of a node's liveliness
    const std::chrono::seconds _max_ping_interval;

    /// Duration at which a node is declared dead if it hasn't responded to pings
    const std::chrono::seconds _ping_dead_threshold;

    /// array of sending socket descriptors
    std::vector<SrcAddrSocket> src_sockets;

    /// destination addresses and associated sending sockets, indexed by IP @c s_addr
    std::map<uint32_t, DestIpSocket> dest_sockets;

    /// stream destinations (references to @p dest_sockets, because a single destination can be used
    /// for multiple streams)
    std::vector<std::reference_wrapper<DestIpSocket>> stream_dest;

    /// initialize sockets used to send data to FRB nodes
    int initialize_source_sockets();

    /// initialize destination addresses and determine the sending socket to use
    int initialize_destinations();

    /// background thread that periodically pings destination hosts and updates their @c live status
    void ping_destinations();

    /// used by @p ping_destinations for periodic sleep interruptible by the @p main_thread on
    /// Kotekan stop
    std::condition_variable ping_cv;
};

#endif
