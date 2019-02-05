#ifndef HSA_PRESUM_KERNEL_H
#define HSA_PRESUM_KERNEL_H

#include "hsaSubframeCommand.hpp"

class hsaPresumKernel : public hsaSubframeCommand {
public:
    hsaPresumKernel(kotekan::Config& config, const string& unique_name,
                    kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaPresumKernel();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t presum_len;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
};

#endif