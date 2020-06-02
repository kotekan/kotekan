/*****************************************
@file
@brief Extract a subset of products from a VisBuffer.
- prodSubset : public kotekan::Stage
*****************************************/
#ifndef PROD_SUB_HPP
#define PROD_SUB_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp" // for dset_id_t, state_id_t, fingerprint_t
#include "visUtil.hpp"        // for prod_ctype

#include <map>      // for map
#include <stddef.h> // for size_t
#include <string>   // for string
#include <tuple>    // for tuple
#include <utility>  // for pair
#include <vector>   // for vector


/**
 * @class prodSubset
 * @brief ``kotekan::Stage`` that extracts a subset of the products.
 *
 * This task consumes a full set of visibilities from a ``VisBuffer`` and
 * passes on a subset of products to an output ``VisBuffer``. The subset
 * extracted depends on the parameter 'prod_subset_type'. Here is a list of
 * values 'prod_subset_type' can take and the parameters they support:
 * - 'all': no extra arameters needed
 *   - Default. All the products.
 * - 'autos': no extra arameters needed
 *   - The subset are all the auto-correlations.
 * - 'baseline': max_ew_baseline, max_ns_baseline
 *   - Selects a subset of products whose baseline length is smaller than the
 *   maximum values given by config parameters in the EW and NS directions.
 * - 'have_inputs': input_list
 *   - The subset are all the correlations containing at leat one of the
 *   inputs in the input_list.
 * - 'only_inputs': input_list
 *   - The subset are all the correlations containing only inputs from the
 *   input_list.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer from which the visibilities are read, can be any size.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 * @buffer out_buf The kotekan buffer which will be fed the subset of visibilities.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 *
 * @conf  prod_subset_type      string. Type of product subset to perform.
 * @conf  num_elements          int. The number of elements (i.e. inputs) in the
 *                              correlator data
 * @conf  num_ev                int. The number of eigenvectors to be stored
 * @conf  max_ew_baseline       int. The maximum baseline length along the EW
 *                              direction to include in subset (in units of the
 *                              shortest EW baseline)
 * @conf  max_ns_baseline       int. The maximum baseline length along the NS
 *                              direction to include in subset (in units of the
 *                              shortest NS baseline)
 * @conf  input_list            vector of int. The list of inputs to include.
 *
 *
 * @warning This will only work correctly if the full correlation triangle is
 * passed in as input.
 *
 * @author  Tristan Pinsonneault-Marotte and Mateus Fandino
 *
 */
class prodSubset : public kotekan::Stage {

public:
    /// Constructor. Loads config options. Defines subset of products.
    prodSubset(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Primary loop: sorts products and passes them on to output buffer.
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

    /// Vector of indices for subset of products (before subsetting on the products in data)
    std::vector<size_t> _base_prod_ind;

    /// Vector of subset of products
    std::vector<prod_ctype> _base_prod_subset;

    // Maps for determining the dataset ID to use
    std::map<dset_id_t, std::pair<dset_id_t, std::vector<size_t>>> dset_id_map;
    std::map<fingerprint_t, std::pair<state_id_t, std::vector<size_t>>> states_map;
};


/**
 * @brief Parse the product subseting section
 * @param config    kotekan::Configuration handle.
 * @param base_path Path into YAML file to search from.
 * @return          Tuple containing a vector of the product inputs, and a
 *                  vector of the corresponding input labels.
 */
std::tuple<std::vector<size_t>, std::vector<prod_ctype>>
parse_prod_subset(kotekan::Config& config, const std::string base_path);


#endif
