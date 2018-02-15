#ifndef HSA_BEAMFORM_TRANSPOSE_H
#define HSA_BEAMFORM_TRANSPOSE_H

#include "hsaCommand.hpp"


class hsaBeamformTranspose: public hsaCommand
{
public:
    hsaBeamformTranspose(Config &config, const string &unique_name,
                         bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaBeamformTranspose();

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:
    int32_t beamform_frame_len;
    int32_t output_frame_len;

    int32_t _num_elements;
    int32_t _samples_per_data_set;
};
REGISTER_HSA_COMMAND(hsaBeamformTranspose);

#endif
