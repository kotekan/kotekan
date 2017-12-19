#ifndef HSA_BEAMFORM_TRANSPOSE_H
#define HSA_BEAMFORM_TRANSPOSE_H

#include "hsaCommand.hpp"


class hsaBeamformTranspose: public hsaCommand
{
public:
    hsaBeamformTranspose(const string &kernel_name, const string &kernel_file_name,
                         hsaDeviceInterface &device, Config &config,
			 bufferContainer &host_buffers,
			 const string &unique_name);

    virtual ~hsaBeamformTranspose();

    void apply_config(const uint64_t& fpga_seq) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t beamform_frame_len;
    int32_t output_frame_len;

    int32_t _num_elements;
    int32_t _samples_per_data_set;
};

#endif
