#ifndef FRBNETWORKPROCESS_HPP
#define FRBNETWORKPROCESS_HPP
 
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <string>
 
class frbNetworkProcess : public KotekanProcess {
public:
  frbNetworkProcess(Config& config,
  const string& unique_name,
  bufferContainer &buffer_container);
  virtual ~frbNetworkProcess();
  void apply_config(uint64_t fpga_seq) override;
  void parse_host_name();
  void main_thread();
private:
  struct Buffer *frb_buf;
  int udp_packet_size;  
  int udp_port_number;
  std::string file_name;
  std::string my_ip_address;
  int number_of_nodes;
  int packets_per_stream;
  int my_node_id;
  char *my_host_name;
};
 
#endif

