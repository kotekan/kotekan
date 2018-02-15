#ifndef HSA_OUTPUT_DATA_ZERO_H
#define HSA_OUTPUT_DATA_ZERO_H

#include "hsaCommand.hpp"

class hsaOutputDataZero: public hsaCommand
{
public:

    hsaOutputDataZero(Config &config, const string &unique_name, 
                  bufferContainer &host_buffers, hsaDeviceInterface &device);

    virtual ~hsaOutputDataZero();

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq,
                         hsa_signal_t precede_signal) override;

private:

    int32_t output_len;
    void * output_zeros;

    // TODO maybe factor these into a CHIME command object class?
    int32_t _num_blocks;
};

#endif