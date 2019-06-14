#ifndef HSA_HFB_ZERO_DATA_H
#define HSA_HFB_ZERO_DATA_H

#include "hsaSubframeCommand.hpp"

class hsaBeamformHFBZeroData : public hsaSubframeCommand {
public:
    hsaBeamformHFBZeroData(kotekan::Config& config, const string& unique_name,
                      kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaBeamformHFBZeroData();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    int32_t output_len;
    void* output_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_blocks;
};

#endif
