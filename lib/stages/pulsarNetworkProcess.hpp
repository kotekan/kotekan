/**
 * @file
 * @brief Network transmission stage for Pulsar obs
 *  - pulsarNetworkProcess : public kotekan::Stage
 */

#ifndef PULSARNETWORKPROCESS_HPP
#define PULSARNETWORKPROCESS_HPP


#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <string> // for string

/**
 * @class pulsarNetworkProcess
 * @brief pulsarNetworkProcess Network transmission stage for Pulsar obs
 *
 *
 * This is an Kotekan stage that collects packetized data from the pulsarPostProcess and
 * transmits 10 beams from pulsarPostProcess to 10 links of pulsar backend.
 * pulsarNetworkProcess distributes the out going traffic to two VLANS (10.15 & 10.16 ) of single 1
 *Gig port. The total pulsar data rate is ~0.26 gbps. The node IP address is derived by parsing the
 *hostname.
 *
 * @par Buffers
 * @buffer in_buf The kotkean buffer to hold the packets to be transmitted to pulsar nodes
 * 	@buffer_format Array of unsigned char.
 * 	@buffer_metadata none
 *
 * @conf   udp_pulsar_packet_size  Int (default 6288). packet size including header
 * @conf   udp_pulsar_port_number  Int (default 1414). udp Port number for pulsar streams
 * @conf   number_of_nodes      Int (default 256). Number of L0 nodes
 * @conf   number_of_subnets    Int (default 2). Number of subnets or VLANS used for transmission of
 *PULSAR data
 * @conf   my_node_id           Int (parsed from the hostname) esimated from the location of node
 *from node location.
 *
 * @todo   Resolve the issue of NTP clock vs Monotonic clock.
 * @todo   Should run further tests
 *
 * @author Arun Naidu
 *
 *
 **/

class pulsarNetworkProcess : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    pulsarNetworkProcess(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);


    /// Destructor , cleaning local allocations
    virtual ~pulsarNetworkProcess();

    /// main thread
    void main_thread() override;

private:
    /// pointer to Input Pulsar buffer
    Buffer* in_buf;

    /// pulsar packet size
    int udp_pulsar_packet_size;

    /// port number
    int udp_pulsar_port_number;

    /// node ip addresses
    char** my_ip_address;

    /// number of L0 nodes
    int number_of_nodes;

    /// number of pulsar VLANS
    int number_of_subnets;

    /// host name from the gethosename()
    char* my_host_name;

    /// samples per packet
    int timesamples_per_pulsar_packet;

    /// packets per stream in a buffer frame
    int num_packet_per_stream;

    /// local socket file descriptor
    int* sock_fd;

    /// array of remote endpoint addresses
    struct sockaddr_in* server_address;

    /// array of local endpoint addresses
    struct sockaddr_in* myaddr;

    /// array of socket ids
    int* socket_ids;

    /// Number of tracking (pulsar) beams
    int _num_pulsar_beams;
};

#endif
