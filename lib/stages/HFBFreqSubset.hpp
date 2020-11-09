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
class HFBFreqSubset : public kotekan::Stage {

public:
    /// Default constructor
    HFBFreqSubset(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Main loop for the stage
    void main_thread() override;

private:
    /// adds state and dataset and gets a new output dataset ID from manager
    void change_dataset_state(dset_id_t input_dset_id);

    // List of frequencies for the subset
    std::vector<uint32_t> _subset_list;

    /// Output buffer with subset of frequencies
    Buffer* out_buf;
    /// Input buffer with all frequencies
    Buffer* in_buf;

    // Maps for determining the dataset ID to use
    std::map<dset_id_t, dset_id_t> dset_id_map;
    std::map<fingerprint_t, state_id_t> states_map;
};


#endif
