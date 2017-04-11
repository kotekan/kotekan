#ifndef HSA_BEAMFORM_KERNEL_H
#define HSA_BEAMFORM_KERNEL_H

#include "gpuHSACommand.hpp"

// #define N_ELEM 2048
// #define N_ITER 38400 // 32768
#define FREQ_REF 492.125984252
#define LIGHT_SPEED 3.e8
#define FEED_SEP 0.3048
#define PI 3.14159265
#define FREQ1 450.0

class hsaBeamformKernel: public gpuHSAcommand
{
public:
    hsaBeamformKernel(const string &kernel_name, const string &kernel_file_name,
                        gpuHSADeviceInterface &device, Config &config,
                        bufferContainer &host_buffers);

    virtual ~hsaBeamformKernel();

    void apply_config(const uint64_t& fpga_seq) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t output_frame_len;
    int32_t map_len;
    int32_t coeff_len;

    uint32_t * host_map;
    float * host_coeff;

    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
};

#endif