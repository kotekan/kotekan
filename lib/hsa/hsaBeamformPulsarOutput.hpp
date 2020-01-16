#ifndef HSA_BEAMFORM_PULSAR_OUTPUT_DATA_H
#define HSA_BEAMFORM_PULSAR_OUTPUT_DATA_H

#include <stdint.h>        // for int32_t
#include <string>          // for string

#include "hsa/hsa.h"       // for hsa_signal_t
#include "hsaCommand.hpp"  // for hsaCommand

class hsaDeviceInterface;
namespace kotekan {
class Config;
class bufferContainer;
}  // namespace kotekan
struct Buffer;

class hsaBeamformPulsarOutput : public hsaCommand {
public:
    hsaBeamformPulsarOutput(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaBeamformPulsarOutput();

    int wait_on_precondition(int gpu_frame_id) override;

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

private:
    Buffer* network_buffer;
    Buffer* output_buffer;

    int32_t network_buffer_id;
    int32_t network_buffer_precondition_id;

    int32_t output_buffer_id;
    int32_t output_buffer_precondition_id;
    int32_t output_buffer_excute_id;
};

#endif
