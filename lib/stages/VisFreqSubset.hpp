/*****************************************
@file
@brief Stage for subsetting visibility data by frequency.
- VisFreqSubset : public FreqSubset
*****************************************/
#ifndef VIS_FREQ_SUBSET_HPP
#define VIS_FREQ_SUBSET_HPP

#include "Config.hpp"
#include "FreqSubset.hpp" // for FreqSubset
#include "Stage.hpp"      // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t, state_id_t, fingerprint_t

#include <map>      // for map
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class VisFreqSubset
 * @brief Outputs a VisBuffer stream with a subset of the input frequencies.
 *
 * This class inherits from the FreqSubset base class and outputs a subset of frequencies
 * @author Mateus Fandino
 */
class VisFreqSubset : public FreqSubset {

public:
    /// Default constructor
    VisFreqSubset(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);

protected:
    // Copy dataset ID from input to output frame
    void copy_dataset_id(dset_id_t dataset_id, frameID input_frame_id,
                         frameID output_frame_id) override;

    // Get dataset ID and frequency ID from frame
    std::pair<dset_id_t, uint32_t> get_frame_data(frameID input_frame_id) override;
};


#endif
