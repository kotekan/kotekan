#ifndef HSA_OUTPUT_DATA_H
#define HSA_OUTPUT_DATA_H

#include "hsaCommand.hpp"

class hsaOutputData: public hsaCommand
{
public:

    hsaOutputData(const string &kernel_name, const string &kernel_file_name,
                  hsaDeviceInterface &device, Config &config,
                  bufferContainer &host_buffers, const string &unique_name);

    virtual ~hsaOutputData();

    void wait_on_precondition(int gpu_frame_id) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

private:
    Buffer * network_buffer;
    Buffer * output_buffer;

    int32_t network_buffer_id;
    int32_t output_buffer_id;

    int32_t output_buffer_precondition_id;
    int32_t output_buffer_excute_id;
};

#endif