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

    /**
     * @brief Execute a kernel.  For chordMVPSetup, this only does book-keeping, no computation.
     * @param pipestate     Pipeline state for this GPU frame.
     * @param pre_events    Array of the last events from each cuda stream, indexed by stream
     *                      number.
     **/
    virtual cudaEvent_t execute(cudaPipelineState& pipestate,
                                const std::vector<cudaEvent_t>& pre_events) override;
};

#endif // KOTEKAN_CHORD_MVP_SETUP_HPP
