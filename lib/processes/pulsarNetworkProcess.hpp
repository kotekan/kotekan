/*****************************************
File Contents:
- pulsarNetworkProcess : public KotekanProcess
*****************************************/

/**
 * @file pulsarNetworkProcess.hpp
 * @brief Network transmission process for Pulsar obs
 *  - pulsarNetworkProcess : public KotekanProcess
 */

#ifndef PULSARNETWORKPROCESS_HPP
#define PULSARNETWORKPROCESS_HPP

 
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <string>


/**
 * @class pulsarNetworkProcess
 * @brief pulsarNetworkProcess Network transmission process for Pulsar obs
 *
 *
 * This is an Kotekan process that collects packetized data from the pulsarPostProcess and 
 * transmits 10 beams from pulsarPostProcess to 10 links of pulsar backend. 
 * pulsarNetworkProcess distributes the out going traffic to two VLANS (10.15 & 10.16 ) of single 1 Gig port.
 * The total pulsar data rate is ~0.26 gbps.
 * The node IP address is derived by parsing the hostname. 
 *
 * @par Buffers
 * @buffer in_buf The kotkean buffer to hold the packets to be transmitted to pulsar nodes
 * 	@buffer_format Array of unsigned char.
 * 	@buffer_metadata none
 *
 * @conf   udp_pulsar_packet_size  Int (default 6288). packet size including header
 * @conf   udp_pulsar_port_number  Int (default 1414). udp Port number for pulsar streams
 * @conf   number_of_nodes      Int (default 256). Number of L0 nodes
 * @conf   number_of_subnets    Int (default 2). Number of subnets or VLANS used for transmission of PULSAR data
 * @conf   my_node_id           Int (parsed from the hostname) esimated from the location of node from node location. 
 *
 * @todo   Resolve the issue of NTP clock vs Monotonic clock. 
 * @todo   Should run further tests 
 *
 * @author Arun Naidu
 *
 *
**/
 
class pulsarNetworkProcess : public KotekanProcess {
public:

  /// Constructor, also initializes internal variables from config.
  pulsarNetworkProcess(Config& config,
  const string& unique_name,
  bufferContainer &buffer_container);
  
  
  /// Destructor , cleaning local allocations
  virtual ~pulsarNetworkProcess();
  
  /// Applies the config parameters
  void apply_config(uint64_t fpga_seq) override;
  
  /// parse hostname to derive the ip_address using gethosname()
  void parse_host_name();

  /// function to add nano seconds to timespec useful for flow control purpose
  void add_nsec(struct timespec &temp, long nsec);

  /// main thread
  void main_thread();
private:

  /// pointer to Input Pulsar buffer 
  struct Buffer *in_buf;

  /// pulsar packet size
  int udp_pulsar_packet_size; 

  /// port number 
  int udp_pulsar_port_number;

  /// node ip addresses
  std::string my_ip_address[2];
  
  /// number of L0 nodes
  int number_of_nodes;

  /// number of pulsar VLANS
  int number_of_subnets;

  /// node id derived from the hostname 
  int my_node_id;

  /// host name from the gethosename()
  char *my_host_name;
};
 
#endif


