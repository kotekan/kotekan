#ifndef GPU_BEAMFORM_PULSAR_SIMULATE_HPP
#define GPU_BEAMFORM_PULSAR_SIMULATE_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"

class gpuBeamformPulsarSimulate : public KotekanProcess {
public:
    gpuBeamformPulsarSimulate(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container);
    ~gpuBeamformPulsarSimulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;
private:
    struct Buffer *input_buf;
    struct Buffer *output_buf;

    // Config options
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _num_pulsar;
    int32_t _num_pol;

    // Unpacked data
    double * input_unpacked;
    unsigned char * cpu_output;
    double * phase;

    int input_len;
    int output_len;
 
    void cpu_beamform_pulsar(double *input, double *phase, unsigned char *output, int nsamp, int nelem, int npsr, int npol);

};

#endif
