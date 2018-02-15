#ifndef HSA_RFI_H
#define HSA_RFI_H

#include "hsaCommand.hpp"

class hsaRfi: public hsaCommand
{
public:
    hsaRfi(Config &config,const string &unique_name,
            bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaRfi();

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t mean_len;
    bool rfi_zero;

    float * Mean_Array;

    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _sk_step;
    int32_t rfi_sensitivity;
};
REGISTER_HSA_COMMAND(hsaRfi);

#endif
