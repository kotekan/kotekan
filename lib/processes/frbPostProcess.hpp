#ifndef FRB_POST_PROCESS
#define FRB_POST_PROCESS

#include "KotekanProcess.hpp"
#include "frb_functions.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"

using std::vector;

class frbPostProcess : public KotekanProcess {
public:
    frbPostProcess(Config& config_,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    virtual ~frbPostProcess();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

private:
    void write_header(unsigned char * dest);

    struct Buffer **in_buf;
    struct Buffer *frb_buf;

    struct FRBHeader frb_header;

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
    bool    _incoherent_beam;

    // Derived useful things
    int32_t num_L1_streams;
    uint32_t num_samples;
    int32_t udp_packet_size;
    int32_t udp_header_size;
    int16_t fpga_counts_per_sample;

};

#endif
