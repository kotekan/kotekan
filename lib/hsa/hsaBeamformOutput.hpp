#ifndef HSA_BEAMFORM_OUTPUT_DATA_H
#define HSA_BEAMFORM_OUTPUT_DATA_H

#include "gpuHSACommand.hpp"

class hsaBeamformOutputData: public gpuHSAcommand
{
public:

    hsaBeamformOutputData(const string &kernel_name, const string &kernel_file_name,
                  gpuHSADeviceInterface &device, Config &config,
                  bufferContainer &host_buffers);

    virtual ~hsaBeamformOutputData();

    void wait_on_precondition(int gpu_frame_id) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

private:
    Buffer * output_buffer;

    int32_t output_buffer_id;

    int32_t output_buffer_precondition_id;
    int32_t output_buffer_excute_id;
};

#endif