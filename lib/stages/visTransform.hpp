/*****************************************
@file
@brief Stage for merging GPU buffers into a visBuffer stream.
- visTransform : public kotekan::Stage
*****************************************/
#ifndef VISTRANSFORM_H
#define VISTRANSFORM_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t
#include "visUtil.hpp"        // for input_ctype, prod_ctype, freq_ctype (ptr only)

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector


/**
 * @class visTransform
 * @brief Merge a set of GPU buffers into a single visBuffer stream.
 *
 * This task takes data coming out of a collecton of GPU streams and merges and
 * reformats it into a single stream in the new visBuffer format that is used
 * for the receiver.
 *
 * @par Buffers
 * @buffer in_bufs The set of buffers coming out the GPU buffers
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf The merged and transformed buffer
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf  num_elements      Int. The number of elements (i.e. inputs) in the
 *                          correlator data.
 * @conf  block_size        Int. The block size of the packed data.
 * @conf  num_ev            Int. The number of eigenvectors to be stored
 * @conf  input_reorder     Array of [int, int, string]. The reordering mapping.
 *                          Only the first element of each sub-array is used and
 *                          it is the the index of the input to move into this
 *                          new location. The remaining elements of the subarray
 *                          are for correctly labelling the input in
 *                          ``Writer``.
 *
 * @author Richard Shaw
 */
class visTransform : public kotekan::Stage {

public:
    // Default constructor
    visTransform(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    // Main loop for the stage
    void main_thread() override;

private:
    // Parameters saved from the config files
    size_t num_elements, num_eigenvectors, block_size;

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<Buffer*, unsigned int>> in_bufs;
    Buffer* out_buf;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

    // dataset ID written to output frames
    dset_id_t _ds_id_out;

    /// Sets the metadataState with a hardcoded weight type ("none"),
    /// prodState, inputState and freqState according to config
    dset_id_t change_dataset_state();

    // data saved to register dataset states
    std::string _instrument_name;
    std::vector<std::pair<uint32_t, freq_ctype>> _freqs;
    std::vector<input_ctype> _inputs;
    std::vector<prod_ctype> _prods;
};

#endif // VISTRANSFORM_H
