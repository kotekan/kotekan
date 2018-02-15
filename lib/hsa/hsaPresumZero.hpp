#ifndef HSA_PRESUM_ZERO_H
#define HSA_PRESUM_ZERO_H

#include "hsaCommand.hpp"

class hsaPresumZero: public hsaCommand
{
public:

    hsaPresumZero(Config &config, const string &unique_name,
                  bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaPresumZero();

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:

    int32_t presum_len;

    void * presum_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_local_freq;
    int32_t _num_elements;
};
REGISTER_HSA_COMMAND(hsaPresumZero);

#endif