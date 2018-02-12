#ifndef HSA_RFI_INPUT_SUM_H
#define HSA_RFI_INPUT_SUM_H

#include "hsaCommand.hpp"


class hsaRfiInputSum: public hsaCommand
{
public:
    hsaRfiInputSum(const string &kernel_name, const string &kernel_file_name,
                        hsaDeviceInterface &device, Config &config,
                        bufferContainer &host_buffers,
                        const string &unique_name);

    virtual ~hsaRfiInputSum();

    void apply_config(const uint64_t& fpga_seq) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:

    uint32_t input_frame_len;
    uint32_t output_frame_len;

    uint32_t _num_elements;
    uint32_t _num_local_freq;
    uint32_t _samples_per_data_set;

    uint32_t _sk_step;
    uint32_t _num_bad_inputs;
    uint32_t _M;
};

#endif
