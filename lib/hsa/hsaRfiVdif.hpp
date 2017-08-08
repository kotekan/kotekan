#ifndef HSA_RFI_VDIF_H
#define HSA_RFI_VDIF_H

#include "hsaCommand.hpp"


class hsaRfiVdif: public hsaCommand
{
public:
    hsaRfiVdif(const string &kernel_name, const string &kernel_file_name,
                        hsaDeviceInterface &device, Config &config,
                        bufferContainer &host_buffers,
                        const string &unique_name);

    virtual ~hsaRfiVdif();

    void apply_config(const uint64_t& fpga_seq) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t input_frame_len;
    int32_t output_len;
    int32_t mean_len;
  
    float * Mean_Array;

    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _sk_step;
    int32_t rfi_sensitivity;
};

#endif
