/**
 * @file frbNetworkSend.hpp
 * @brief Network transmission stage for FRB obs
 *  - frbNetworkSend : public kotekan::Stage
 */

#ifndef FRBNETWORKSEND_HPP
#define FRBNETWORKSEND_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "Telescope.hpp"       // for Telescope
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
 * @class frbNetworkSend
 * @brief frbNetworkSend Network transmission stage for FRB obs
 *
 *
 * This is an Kotekan stage that read packetized data from frbPostProcess and transmits 1024 beams
 * to 256 links of frb backend. frbNetworkSend distributes the out going traffic to four VLANS
 * (10.6 10.7 10.8 10.9) on single 1 Gig port. The frb total data rate is ~0.55 gbps. The node IP
 * address is derived by parsing the hostname.
 *
 * @par REST Endpoints
 * @endpoint /frb/update_gains/``gpu_id`` Any contact here triggers a re-parse of the gains file.
 * @endpoint /frb/update_destination Set the active status of param ``host`` to value of param
 * ``active``. Data is sent only to active hosts.
 *
 * @par Buffers
 * @buffer in_buf The kotkean buffer to hold the data to be transmitted to L1 nodes
 *      @buffer_format Array of signed char
 *      @buffer_metadata none
 * @buffer offsetscale_buf The kotkean buffer to hold the offset and scale
 *      values for the data blocks to be transmitted to L1 nodes
 *      @buffer_format Array of pairs of float32
 *      @buffer_metadata none
 *
 *
 * @conf   udp_frb_packet_size  Int (default 4264). packet size including header
 * @conf   udp_frb_port_number  Int (default 1313). udp Port number for frb streams
 * @conf   number_of_nodes      Int (default 256). Number of L0 nodes
 * @conf   number_of_subnets    Int (default 4). Number of subnets or VLANS used for transmission of
 * FRB data. Use magic value of 0 for local testing.
 * @conf   packets_per_stream   Int (default 8). Number of packets for each stream within each frame
 * @conf   L1_node_ips          Array of Strings. List of IPs to send to. (?)
 * @conf   beam_offset          Int (default 0). Offset the beam_id going to L1 Process
 * @conf   column_mode          bool (default false) Send beams in a single CHIME cylinder.
 * @conf   ping_interval        Uint32 (default 6 min) Time in seconds between sending a ping to
 * check destination is live
 * @conf   quick_ping_interval  Uint32 (default 5 sec) Time in seconds for sending pings when a live
 * node stops responding
 * @conf   ping_dead_threshold  Uint32 (default 30 sec) Duration in seconds of quick-checking state
 * after which a node is declared dead if it still hasn't responded. If 0, disable the checks
 * entirely.
 *
 * @todo   Resolve the issue of NTP clock vs Monotonic clock.
 *
 * @author Arun Naidu, Davor Cubranic
 *
 */


class frbNetworkSend : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    frbNetworkSend(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);

    /// Destructor , cleaning local allocations
    virtual ~frbNetworkSend();

    /// Callback to update the beam offset
    void update_offset_callback(kotekan::connectionInstance& conn, nlohmann::json& json_request);

    /// Callback to change destination active status
    void set_destination_active_callback(kotekan::connectionInstance& conn,
                                         nlohmann::json& json_request);

    /// main thread
    void main_thread() override;

private:
    struct SrcAddrSocket {
        const sockaddr_in addr;
        const int socket_fd;
    };


    /**
     * @brief Convenience struct used to hold all relevant information about an FRB L1 destination
     */
    struct DestIpSocket {
        /// Regular constructor used with data from the config file
        DestIpSocket(std::string host, sockaddr_in addr, int s, bool active = true) :
            host(std::move(host)), addr(std::move(addr)), sending_socket(s), active(active),
            live(false){};


        /// Move constructor is necessary for inserting into standard containers
        DestIpSocket(DestIpSocket&& other) :
            host(std::move(other.host)), addr(std::move(other.addr)),
            sending_socket(other.sending_socket), active(other.active), live(other.live.load()){};

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

    /// pointer to input beams buffer
    Buffer* in_buf;

    /// pointer to offset and scales buffer
    Buffer* offsetscale_buf;

    /// frb packet size
    int udp_frb_packet_size;

    /// port number
    int udp_frb_port_number;

    /// number of L0 nodes
    int number_of_nodes;

    /// number of VLANS, use magic value of 0 to use 127.0.0.1 only
    int number_of_subnets;

    /// number of packets to each L1 nodes
    int packets_per_stream;

    /// beam offset for 8-node frb system
    int beam_offset;

    // samples per packet
    int samples_per_packet;

    // total number of coarse frequencies in incoming beams buffer. Must be
    // evenly divided by _nfreq_coarse
    int num_frequencies;

    // hard-coded size constraints (due ot packet layout expected by receiver)
    static const int _nfreq_coarse = 4;
    static const int _factor_upchan_out = 16;
    static const int _timesamples_per_frb_packet = 16;
    static const int _nbeams = 4;          // number of beams per FRB network packets
    static const int _total_nbeams = 1024; // number of beams formed by beamformer

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

    // the telescope object to handle conversion to site time
    const Telescope& tel;

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
