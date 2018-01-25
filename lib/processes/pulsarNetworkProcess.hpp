/*****************************************
File Contents:
- pulsarNetworkProcess : public KotekanProcess
*****************************************/


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
 * This is an Kotekan process that transmits 10 beams from pulsarPostProcess to 10 links of pulsar backend. 
 * pulsarNetworkProcess distributes the out going traffic to two VLANS (10.15 10.16 ) of single 1 Gig port.
 * The frb total data rate is ~0.26 gbps.
 * The node IP address is derived by parsing the hostname. 
 *
 * @conf   udp_pulsar_packet_size  Int (default 6288). packet size including header
 * @conf   udp_pulsar_port_number  Int (default 1313). udp Port number for frb streams
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
  
  /// parse config
  void apply_config(uint64_t fpga_seq) override;
  
  /// parse hostname to derive the ip_address uses gethosname()
  void parse_host_name();

  /// function to add nano seconds to timespec useful for packet timming purpose
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


