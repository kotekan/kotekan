#ifndef HSA_SLEEPER_H
#define HSA_SLEEPER_H

#include "hsaCommand.hpp"

class hsaSleeper: public hsaCommand
{
public:

    // Pull constructors.
    using hsaCommand::hsaCommand;

    hsa_signal_t execute(int gpu_frame_id, const uint64_t& fpga_seq, hsa_signal_t precede_signal) override;

    void finalize_frame(int frame_id) override;

    virtual ~hsaSleeper();

};

#endif