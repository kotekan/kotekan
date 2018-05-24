/**
 * @file
 * @brief Network transmission process for FRB obs
 *  - frbNetworkProcess : public KotekanProcess
 */

#ifndef FRBNETWORKPROCESS_HPP
#define FRBNETWORKPROCESS_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"
#include <string>
#include "restServer.hpp"

 /**
 * @class frbNetworkProcess
 * @brief frbNetworkProcess Network transmission process for FRB obs
 *
 *
 * This is an Kotekan process that read packetized data from frbPostProcess and transmits 1024 beams to 256 links of frb backend.
 * frbNetworkProcess distributes the out going traffic to four VLANS (10.6 10.7 10.8 10.9) on single 1 Gig port.
 * The frb total data rate is ~0.55 gbps. The node IP address is derived by parsing the hostname.
 *
 * @par REST Endpoints
 * @endpoint /frb/update_gains/``gpu_id`` Any contact here triggers a re-parse of the gains file.
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
 * @conf   number_of_subnets    Int (default 4). Number of subnets or VLANS used for transmission of FRB data
 * @conf   packets_per_stream   Int (default 8). Number of subnets or VLANS used for transmission of FRB data
 * @conf   L1_node_ips          Array of Strings. List of IPs to send to. (?)
 * @conf   beam_offset          Int (default 0). Offset the beam_id going to L1 Process
 * @conf   time_interval        Unsigned long (default 125829120). Time per buffer in ns.
 * @conf   column_mode          bool (default false) Send beams in a single CHIME cylinder.
 * @todo   Resolve the issue of NTP clock vs Monotonic clock.
 *
 * @author Arun Naidu
 *
 */


class frbNetworkProcess : public KotekanProcess {
public:
  /// Constructor, also initializes internal variables from config.
  frbNetworkProcess(Config& config,
  const string& unique_name,
  bufferContainer &buffer_container);

  /// Destructor , cleaning local allocations
  virtual ~frbNetworkProcess();

  ///parse config
  void apply_config(uint64_t fpga_seq) override;

  /// parse hostname to derive the ip_address using gethosname()
  void parse_host_name();

  /// Callback to update the beam offset
  void update_offset_callback(connectionInstance& conn, json& json_request);

  /// main thread
  void main_thread();
private:

  /// pointer to Input FRB buffer
  struct Buffer *in_buf;

  /// frb packet size
  int udp_frb_packet_size;

  /// port number
  int udp_frb_port_number;

  /// node ip addresses
  std::string my_ip_address[4];

  /// number of L0 nodes
  int number_of_nodes;

  /// number of VLANS
  int number_of_subnets;

  /// number of packets to each L1 nodes
  int packets_per_stream;

  /// node id derived from the hostname
  int my_node_id;

  /// host name from the gethosename()
  char *my_host_name;

  /// beam offset for 8-node frb system
  int beam_offset;

  // time per buffer frame in ns
  unsigned long time_interval;

  //Beam Configuration Mode
  bool column_mode;

  /// The endpoint name for the restServer
  std::string endpoint;
};

#endif


