/*****************************************
@file
@brief Stage for subsetting data by frequency.
- FreqSubset : public kotekan::Stage
*****************************************/
#ifndef FREQ_SUBSET_HPP
#define FREQ_SUBSET_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, state_id_t, fingerprint_t

#include <map>      // for map
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class FreqSubset
 * @brief Outputs a Buffer stream with a subset of the input frequencies.
 *
 * This task takes data coming out of a Buffer stream and selects a subset of
 * frequencies to be passed on to the output buffer.
 *
 * @par Buffers
 * @buffer in_buf The original buffer with all frequencies
 *         @buffer_format Vis/HFBBuffer structured
 *         @buffer_metadata Vis/HFBMetadata
 * @buffer out_buf The buffer containing the subset of frequencies
 *         @buffer_format Vis/HFBBuffer structured
 *         @buffer_metadata Vis/HFBMetadata
 *
 * @conf  subset_list           Vector of Int. The list of frequencies that go
 *                              in the subset.
 *
 * @author Mateus Fandino
 */
template<typename T>
class FreqSubset : public kotekan::Stage {

public:
    /// Default constructor
    FreqSubset(kotekan::Config& config, const std::string& unique_name,
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
