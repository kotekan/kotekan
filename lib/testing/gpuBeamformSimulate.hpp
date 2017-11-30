#ifndef GPU_BEAMFORM_SIMULATE_HPP
#define GPU_BEAMFORM_SIMULATE_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"

class gpuBeamformSimulate : public KotekanProcess {
public:
    gpuBeamformSimulate(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container);
    ~gpuBeamformSimulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *input_buf;
    struct Buffer *output_buf;

    // Config options
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _factor_upchan;
    int32_t _downsample_time;
    int32_t _downsample_freq;
    vector<int32_t> _reorder_map;

    float * coff;

    // Unpacked data
    double * input_unpacked;
    double * input_unpacked_padded;
    double * clamping_output;
    double * cpu_beamform_output;
    double * transposed_output;
    double * tmp128;
    int * tmp512;
    int * reorder_map_c;
    unsigned char * cpu_final_output;


    int input_len;
    int input_len_padded;
    int transposed_len;
    int output_len;

    void reorder(unsigned char *data, int *map);
    void cpu_beamform_ns(double *data, unsigned long transform_length, int stop_level);
    void cpu_beamform_ew(double *input, double *output, float *Coeff, int nbeamsNS, int nbeamsEW, int npol, int nsamp_in);
    void clamping(double *input, double *output, float freq, int nbeamsNS, int nbeamsEW, int nsamp_in, int npol);
    void transpose(double *input, double *output, int nbeams, int nsamp_in);
    void upchannelize(double *data, int upchan);
};

#endif
