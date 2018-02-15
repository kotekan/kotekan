#ifndef HSA_BARRIER_H
#define HSA_BARRIER_H

#include "hsaCommand.hpp"

class hsaBarrier: public hsaCommand
{
public:

    hsaBarrier(const string &kernel_name, const string &kernel_file_name,
                    hsaDeviceInterface &device, Config &config,
                    bufferContainer &host_buffers,
                    const string &unique_name);

    virtual ~hsaBarrier();

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;
};

#endif