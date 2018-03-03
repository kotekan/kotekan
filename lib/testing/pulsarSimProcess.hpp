#ifndef PULSAR_SIM_PROCESS
#define PULSAR_SIM_PROCESS

#include "KotekanProcess.hpp"
#include <vector>

using std::vector;

class pulsarSimProcess : public KotekanProcess {
public:
    pulsarSimProcess(Config& config_,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    virtual ~pulsarSimProcess();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);
    void parse_host_name();
private:
    void fill_headers(unsigned char * out_buf,
                  struct VDIFHeader * vdif_header,
                  const uint64_t fpga_seq_num,
		  struct timeval * time_now,
		  struct psrCoord * psr_coord,
		  uint16_t * freq_ids);
  
    struct Buffer **in_buf;
    struct Buffer *pulsar_buf;

    // Config variables
    int32_t _num_gpus;
    int32_t _samples_per_data_set;
    int32_t _nfreq_coarse;
    int32_t _num_pulsar;
    int32_t _num_pol;
    int32_t _timesamples_per_pulsar_packet;
    int32_t _udp_packet_size;
    int32_t _udp_header_size;
    struct timeval time_now;
     /// node id derived from the hostname 
  int my_node_id;

  /// host name from the gethosename()
  char *my_host_name = new char[20];   
  int number_of_subnets;
   /// node ip addresses
  std::string my_ip_address[2]; 
};

#endif
