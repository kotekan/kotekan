#ifndef HSA_BEAMFORM_KERNEL_H
#define HSA_BEAMFORM_KERNEL_H

#include "hsaCommand.hpp"
#include "restServer.hpp"

// #define N_ELEM 2048
// #define N_ITER 38400 // 32768
#define FREQ_REF 492.125984252
#define LIGHT_SPEED 3.e8
#define FEED_SEP 0.3048
#define PI 3.14159265

class hsaBeamformKernel: public hsaCommand
{
public:
    hsaBeamformKernel(Config &config, const string &unique_name, 
                        bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaBeamformKernel();

    int wait_on_precondition(int gpu_frame_id) override;

    void calculate_cl_index(uint32_t *host_map, float freq1, float *host_coeff, float *_ew_spacing_c);

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

    void update_gains_callback(connectionInstance& conn, json& json_request);
private:
    int32_t input_frame_len;
    int32_t output_frame_len;
    int32_t map_len;
    int32_t coeff_len;
    int32_t gain_len;
    string _gain_dir;
    vector<float> default_gains;

    Buffer * metadata_buf;
    int32_t metadata_buffer_id;
    int32_t metadata_buffer_precondition_id;
    int32_t freq_idx;
    float freq_MHz;

    uint32_t * host_map;
    float * host_coeff;
    float * host_gain;

    float scaling;

    uint32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    bool update_gains; //so gains only load on request!
    bool first_pass; //avoid re-calculating freq-specific params

    vector<float> _ew_spacing;
    float * _ew_spacing_c;
};

#endif
