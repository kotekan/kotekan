#ifndef PULSAR_POST_PROCESS
#define PULSAR_POST_PROCESS

#include "KotekanProcess.hpp"
#include <vector>

using std::vector;

class pulsarPostProcess : public KotekanProcess {
public:
    pulsarPostProcess(Config& config_,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    virtual ~pulsarPostProcess();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

private:
    void fill_headers(unsigned char * out_buf,
                  struct VDIFHeader * vdif_header,
                  const uint64_t fpga_seq_num,
		  const uint32_t gps_time,
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

};

#endif
