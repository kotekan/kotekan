#ifndef N2FRAME_TO_VISFRAME_HPP
#define N2FRAME_TO_VISFRAME_HPP

#include <stdint.h>             // for uint32_t
#include <string>               // for string, basic_string
#include <vector>               // for vector
#include <utility>              // for pair

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "datasetManager.hpp"   // for datasetManager, state_id_t
#include "visUtil.hpp"          // for freq_ctype, input_ctype, prod_ctype

/**
 * @brief Translates a N2FrameView buffer to a VisFrameView buffer by complex
 * conjugating the visibility matrix and quantities derived from the
 * eigenvectors as needed. Also adds dataset ids as if the VisFrameView had been
 * passed along CHIME's hfa pipeline instead.
 *
 * @par Buffers
 * @buffer n2_buf The accumulated and tagged data.
 *         @buffer_format VisBuffer structured.
 *         @buffer_metadata N2Metadata
 * @buffer vis_buf Output buffer.
 *         @buffer_format VisBuffer structured.
 *         @buffer_metadata VisMetadata
 *
 * @conf  n2_buf  Buffers to hold the N2FrameView
 * @conf  vis_buf Buffers to hold the VisFrameView
 * @conf  fake_git_tag String. Fake git hash in visMetadata to make receiver
 *        accept it. Leave empty to use actual git tag.
 *
 * @author Roland Haas
 */
class n2FrameToVisFrame : public kotekan::Stage {
public:
    /// Standard constructor
    n2FrameToVisFrame(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);

    /// Destructor
    ~n2FrameToVisFrame();

    /// Main thead which zeros the data from the lost_samples_buf
    void main_thread() override;

private:
    void register_base_dataset_states(std::string& instrument_name,
                                      std::vector<std::pair<uint32_t, freq_ctype>>& freqs,
                                      std::vector<input_ctype>& inputs,
                                      std::vector<prod_ctype>& prods);

private:
    datasetManager& dm;

    // the base states (input, prod, freq, meta)
    std::vector<state_id_t> base_dataset_states;

    /// The buffer with the output of N2Accumulate
    Buffer* n2_buf;

    /// The target buffer for a visFrame
    Buffer* vis_buf;

    /// Fake git hash to record in visMetadata dataset id
    std::string fake_git_tag;
};

#endif /* N2FRAME_TO_VISFRAME_HPP */
