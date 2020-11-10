/*****************************************
@file
@brief Stage for subsetting visibility data by frequency.
- HFBFreqSubset : public kotekan::Stage
*****************************************/
#ifndef HFB_FREQ_SUBSET_HPP
#define HFB_FREQ_SUBSET_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t, state_id_t, fingerprint_t
#include "FreqSubset.hpp" // for FreqSubset

#include <map>      // for map
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class HFBFreqSubset
 * @brief Outputs a VisBuffer stream with a subset of the input frequencies.
 *
 * This task takes data coming out of a VisBuffer stream and selects a subset of
 * frequencies to be passed on to the output buffer.
 *
 * @par Buffers
 * @buffer in_buf The original buffer with all frequencies
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 * @buffer out_buf The buffer containing the subset of frequencies
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf  subset_list           Vector of Int. The list of frequencies that go
 *                              in the subset.
 *
 * @author Mateus Fandino
 */
class HFBFreqSubset : public FreqSubset {

public:
    /// Default constructor
    HFBFreqSubset(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

protected:
    // Copy dataset ID from input to output frame
    void copy_dataset_id(dset_id_t dataset_id, frameID input_frame_id, frameID output_frame_id) override;

    // Get dataset ID and frequency ID from frame
    std::pair<dset_id_t, uint32_t> get_frame_data(frameID input_frame_id) override;
};


#endif
