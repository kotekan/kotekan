#ifndef GPU_BEAMFORM_SIMULATE_HPP
#define GPU_BEAMFORM_SIMULATE_HPP

#include "buffers.h"
#include "KotekanProcess.hpp"

class gpuBeamformSimulate : public KotekanProcess {
public:
    gpuBeamformSimulate(Config &config,
                const string& unique_name,
                struct Buffer &input_buf,
                struct Buffer &output_buf);
    ~gpuBeamformSimulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer &input_buf;
    struct Buffer &output_buf;

    // Config options
    int32_t _num_elements;
    int32_t _samples_per_data_set;

    float * coff;

    // Unpacked data
    double * input_unpacked;
    double * input_unpacked_padded;
    double * clamping_output;
    double * final_output;

    int input_len;
    int input_len_padded;
    int clamping_output_len;
    int final_output_len;

    void cpu_beamform_ns(double *data, unsigned long transform_length, int stop_level);
    void cpu_beamform_ew(double *input, double *output, float *Coeff, int nbeamsNS, int nbeamsEW, int npol, int nsamp_in);
    void clamping(double *input, double *output, float freq, int nbeamsNS, int nbeamsEW, int nsamp_in, int npol);
};

#endif