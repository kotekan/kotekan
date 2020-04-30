/*****************************************
@file
@brief Extract a subset of inputs from a visBuffer.
- inputSubset : public kotekan::Stage
*****************************************/
#ifndef INPUT_SUBSET_HPP
#define INPUT_SUBSET_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, state_id_t, fingerprint_t

#include <map>      // for map
#include <set>      // for set
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


/**
 * @class InputSubset
 * @brief ``kotekan::Stage`` that extracts a subset of the inputs.
 *
 * The difference between this stage and the only_inputs option of prodSubset
 * is that this will also subset the `num_elements` sized parts of the frames,
 * such as gains and eigenvectors.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer from which the visibilities are read, can be any size.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 * @buffer out_buf The kotekan buffer which will be fed the subset of visibilities.
 *     @buffer_format visBuffer structured
 *     @buffer_metadata visMetadata
 *
 * @conf  inputs  List of ints. The channel IDs for the inputs to extract.
 *
 * @author Richard Shaw
 */
class InputSubset : public kotekan::Stage {

public:
    /// Constructor. Loads config options.
    InputSubset(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);

    /// Primary loop: extracts inputs and passes them on to output buffer.
    void main_thread() override;

private:
    /// keeps track of the input dataset ID
    /// and gets new output dataset ID from manager
    ///
    void change_dataset_state(dset_id_t ds_id);

    /// Input buffer
    Buffer* in_buf;

    /// Output buffer to receive baseline subset visibilities
    Buffer* out_buf;

    /// Vector of indices for inputs
    const std::set<uint32_t> _inputs;

    // Cache of the indices we need to select from the incoming data
    std::vector<uint32_t> prod_ind;
    std::vector<uint32_t> input_ind;

    // Fingerprint of the input and product states for checking they haven't changed
    fingerprint_t fingerprint = fingerprint_t::null;

    // Map known input dataset_id's to the output ID we should use
    std::map<dset_id_t, dset_id_t> dset_id_map;

    // The replacement input and product states to attach
    std::vector<state_id_t> states;
};

#endif
