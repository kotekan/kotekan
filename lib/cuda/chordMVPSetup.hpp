#ifndef KOTEKAN_CHORD_MVP_SETUP_HPP
#define KOTEKAN_CHORD_MVP_SETUP_HPP

#include "cudaCommand.hpp"

#include <vector>

/**
 * @class chordMVPSetup
 * @brief A CUDA GPU command for setting up buffers for our Minimum Viable Pipeline demo.
 *
 * @author Dustin Lang
 */
class chordMVPSetup : public cudaCommand {
public:
    chordMVPSetup(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~chordMVPSetup();
    virtual cudaEvent_t execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                                bool* quit) override;
};

#endif // KOTEKAN_CHORD_MVP_SETUP_HPP
