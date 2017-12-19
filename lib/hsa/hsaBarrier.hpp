#ifndef HSA_BARRIER_H
#define HSA_BARRIER_H

#include "hsaCommand.hpp"

class hsaBarrier: public hsaCommand
{
public:

    // Pull constructors.
    using hsaCommand::hsaCommand;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

    virtual ~hsaBarrier();

};

#endif