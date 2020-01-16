#ifndef HSA_PRESUM_ZERO_H
#define HSA_PRESUM_ZERO_H

#include <stdint.h>                // for int32_t
#include <string>                  // for string

#include "hsa/hsa.h"               // for hsa_signal_t
#include "hsaSubframeCommand.hpp"  // for hsaSubframeCommand

class hsaDeviceInterface;
namespace kotekan {
class Config;
class bufferContainer;
}  // namespace kotekan

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
