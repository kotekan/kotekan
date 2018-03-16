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
    void main_thread() override;
    virtual void apply_config(uint64_t fpga_seq) override;

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
    uint32_t _num_gpus;
    uint32_t _samples_per_data_set;
    uint32_t _nfreq_coarse;
    uint32_t _num_pulsar;
    uint32_t _num_pol;
    uint32_t _timesamples_per_pulsar_packet;
    uint32_t _udp_packet_size;
    uint32_t _udp_header_size;
    struct timeval time_now;
};

#endif
