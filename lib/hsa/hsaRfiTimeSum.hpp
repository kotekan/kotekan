#ifndef HSA_RFI_TIME_SUM_H
#define HSA_RFI_TIME_SUM_H

#include "hsaCommand.hpp"

class hsaRfi: public hsaCommand
{
public:
    hsaRfi(Config &config,const string &unique_name,
            bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaRfiTimeSum();

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    uint32_t input_frame_len;
    uint32_t output_frame_len;
    uint32_t mask_len;
    
    uint8_t *InputMask;

    uint32_t _num_elements;
    uint32_t _num_local_freq;
    uint32_t _samples_per_data_set;

    uint32_t _sk_step;
    uint32_t _num_bad_inputs;
};

#endif
