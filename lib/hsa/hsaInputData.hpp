#ifndef HSA_INPUT_DATA_H
#define HSA_INPUT_DATA_H

#include "Config.hpp"             // for Config
#include "buffer.hpp"             // for Buffer
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

class hsaInputData : public hsaCommand {
public:
    hsaInputData(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaInputData();

    int wait_on_precondition(int gpu_frame_id) override;

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

private:
    int32_t network_buffer_id;
    int32_t network_buffer_precondition_id;
    int32_t network_buffer_finalize_id;
    Buffer* network_buf;
    size_t input_frame_len;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    float _delay_max_fraction;

    // Random delay in seconds
    double _random_delay;

    // Apply a random delay to spread out the power load if set to true.
    bool _enable_delay;

    const double _sample_arrival_rate = 390625.0;
};

#endif
