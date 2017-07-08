#ifndef NETWORK_OUTPUT_SIM
#define NETWORK_OUTPUT_SIM

#define SIM_CONSTANT   0
#define SIM_FULL_RANGE 1
#define SIM_SINE        2

#include "buffer.c"
#include "errors.h"
#include "KotekanProcess.hpp"

class networkOutputSim : public KotekanProcess {
public:
    networkOutputSim(Config &config,
                     const string& unique_name,
                     bufferContainer &buffer_container);
    virtual ~networkOutputSim();
    void main_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer * buf;
    int num_links_in_group;
    int link_id;
    int pattern;
    int stream_id;

    // Config variables.
    int32_t _samples_per_data_set;
    int32_t _num_local_freq;
    int32_t _num_elem;

};

#endif