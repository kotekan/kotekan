#ifndef HSA_BARRIER_H
#define HSA_BARRIER_H

#include "hsa/hsa.h"      // for hsa_signal_t
#include "hsaCommand.hpp" // for hsaCommand

#include <string> // for string

class hsaDeviceInterface;
namespace kotekan {
class Config;
class bufferContainer;
} // namespace kotekan

class hsaBarrier : public hsaCommand {
public:
    hsaBarrier(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaBarrier();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;
};

#endif
