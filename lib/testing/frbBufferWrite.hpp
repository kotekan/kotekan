#ifndef FRBBUFFERWRITE_HPP
#define FRBBUFFERWRITE_HPP

#include "KotekanProcess.hpp"
//#include "Config.hpp"
//#include "buffers.h"
#include <vector>
//#include "fpga_header_functions.h"

using std::vector;

class frbBufferWrite : public KotekanProcess {
public:
    frbBufferWrite(Config& config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    virtual ~frbBufferWrite();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);
    void copyShort(unsigned char *buf, uint16_t* data, int num_elements);
    void copyFloat(unsigned char *buf, float* data, int num_elements);
    int random_generator(float *intensity,int size);

private:
    void fill_headers(unsigned char * out_buf,
                  struct FRBHeader * frb_header,
                  const uint64_t fpga_seq_num,
		  const uint16_t num_L1_streams,
		  uint16_t * frb_header_coarse_freq_ids,
		  float * frb_header_scale,
		  float * frb_header_offset);

    struct Buffer **in_buf;
    struct Buffer *frb_buf;

    //Dynamic header
    uint16_t * frb_header_beam_ids;
    uint16_t * frb_header_coarse_freq_ids;
    float * frb_header_scale;
    float * frb_header_offset;

    // Config variables
    int32_t _num_gpus;
    int32_t _samples_per_data_set;
    int32_t _nfreq_coarse;
    int32_t _downsample_time;
    int32_t _factor_upchan;
    int32_t _factor_upchan_out;
    int32_t _nbeams;
    int32_t _timesamples_per_frb_packet;
    int32_t _udp_frb_packet_size;
    int32_t _udp_frb_header_size;
    int16_t _fpga_counts_per_sample;
    vector<int32_t> _freq_array;
    int my_node_id;
    std::string my_ip_address;
    char *my_host_name;
  //uint16_t encode_stream_id(const stream_id_t s_stream_id);
  // stream_id_t extract_stream_id(const uint16_t encoded_stream_id);


};

#endif
