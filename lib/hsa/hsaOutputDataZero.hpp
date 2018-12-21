#ifndef HSA_OUTPUT_DATA_ZERO_H
#define HSA_OUTPUT_DATA_ZERO_H

#include "hsaSubframeCommand.hpp"

class hsaOutputDataZero : public hsaSubframeCommand {
public:
    hsaOutputDataZero(Config& config, const string& unique_name, bufferContainer& host_buffers,
                      hsaDeviceInterface& device);

    virtual ~hsaOutputDataZero();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    int32_t output_len;
    void* output_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_blocks;
};

#endif