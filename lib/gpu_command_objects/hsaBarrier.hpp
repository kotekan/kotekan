#ifndef HSA_BARRIER_H
#define HSA_BARRIER_H

#include "gpuHSACommand.hpp"

class hsaBarrier: public gpuHSAcommand
{
public:

    // Pull constructors.
    using gpuHSAcommand::gpuHSAcommand;

    void wait_on_precondition(int gpu_frame_id) override;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

    virtual ~hsaBarrier();

};

#endif