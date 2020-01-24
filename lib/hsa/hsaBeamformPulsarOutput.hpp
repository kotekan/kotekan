#ifndef HSA_BEAMFORM_PULSAR_OUTPUT_DATA_H
#define HSA_BEAMFORM_PULSAR_OUTPUT_DATA_H

#include "Config.hpp"             // for Config
#include "buffer.h"               // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

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
