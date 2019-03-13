#ifndef HSA_PRESUM_ZERO_H
#define HSA_PRESUM_ZERO_H

#include "hsaSubframeCommand.hpp"

class hsaPresumZero : public hsaSubframeCommand {
public:
    hsaPresumZero(kotekan::Config& config, const string& unique_name,
                  kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaPresumZero();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    int32_t presum_len;

    void* presum_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;
};

#endif