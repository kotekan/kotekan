#ifndef HSA_PRESEED_KERNEL_H
#define HSA_PRESEED_KERNEL_H

#include "hsaCommand.hpp"

class hsaPreseedKernel: public hsaCommand
{
public:

    hsaPreseedKernel(const string &kernel_name, const string &kernel_file_name,
                  hsaDeviceInterface &device, Config &config,
                  bufferContainer &host_buffers);

    virtual ~hsaPreseedKernel();

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                            hsa_signal_t precede_signal) override;

    void apply_config(const uint64_t& fpga_seq) override;

private:

    int32_t input_frame_len;
    int32_t presum_len;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
};

#endif