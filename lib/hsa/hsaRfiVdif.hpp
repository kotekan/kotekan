#ifndef HSA_RFI_VDIF_H
#define HSA_RFI_VDIF_H

#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

class hsaRfiVdif : public hsaCommand {
public:
    hsaRfiVdif(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    virtual ~hsaRfiVdif();

    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t output_len;
    int32_t mean_len;

    float* Mean_Array;

    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _sk_step;
    int32_t rfi_sensitivity;
};

#endif
