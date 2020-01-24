#ifndef HSA_PRESUM_ZERO_H
#define HSA_PRESUM_ZERO_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface
#include "hsaSubframeCommand.hpp" // for hsaSubframeCommand

#include <stdint.h> // for int32_t
#include <string>   // for string

class hsaPresumZero : public hsaSubframeCommand {
public:
    hsaPresumZero(kotekan::Config& config, const std::string& unique_name,
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
