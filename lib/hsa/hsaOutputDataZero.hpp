#ifndef HSA_OUTPUT_DATA_ZERO_H
#define HSA_OUTPUT_DATA_ZERO_H

#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaSubframeCommand.hpp" // for hsaSubframeCommand

#include <stdint.h> // for int32_t
#include <string>   // for string

class hsaDeviceInterface;
namespace kotekan {
class Config;
class bufferContainer;
} // namespace kotekan

class hsaOutputDataZero : public hsaSubframeCommand {
public:
    hsaOutputDataZero(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaOutputDataZero();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    int32_t output_len;
    void* output_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_blocks;
};

#endif
