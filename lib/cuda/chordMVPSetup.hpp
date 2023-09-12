#ifndef KOTEKAN_CHORD_MVP_SETUP_HPP
#define KOTEKAN_CHORD_MVP_SETUP_HPP

#include "cudaCommand.hpp"

#include <vector>

/**
 * @class chordMVPSetup
 * @brief A CUDA GPU command for setting up buffers for our Minimum Viable Pipeline demo.
 *
 * Currently, this only sets up a couple of GPU buffer views to hack
 * our way around shortcomings in the MVP-era prototype code.
 *
 * Specifically:
 * - cudaUpchannelize produces outputs of size 2048.
 * - cudaFRBBeamformer consumes inputs of multiples of 48.
 * These don't match, so the cudaFRBBeamformer code manages a buffer
 * of leftover data that will get processed with the next frame of
 * data.  We want cudaUpchannelize to always be writing into a fixed
 * output array, so we make that an offset view into the buffer that
 * cudaFRBBeamformer will use for its input.  It then copies its
 * leftover bit of data from last time just before the
 * cudaUpchannelize output, computes the correct starting address in
 * the input buffer, and calls the GPU kernel with that pointer.
 *
 * - we use a buffer view to create an array of the right size (but
 * not the right contents!) to feed to an upchannelizer and create the
 * "Fine Visibility" matrix.  (We should subset the frequencies, but
 * for MVP purposes we just need to test the rate.  The data layout
 * has T varying most slowly, so we can do a time subset just with
 * array views, while doing the Frequency subset would require a
 * cudaMemcpy3DAsync
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

    virtual std::string get_extra_dot(const std::string& prefix) const override;
};

#endif // KOTEKAN_CHORD_MVP_SETUP_HPP
