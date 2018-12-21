/*****************************************
@file
@brief Extract a subset of products from a visBuffer.
- prodSubset : public KotekanProcess
*****************************************/
#ifndef PROD_SUB_HPP
#define PROD_SUB_HPP

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp"
#include "visUtil.hpp"

#include <future>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <tuple>
#include <vector>


/**
 * @class prodSubset
 * @brief ``KotekanProcess`` that extracts a subset of the products.
 *
 * This task consumes a full set of visibilities from a ``visBuffer`` and
 * passes on a subset of products to an output ``visBuffer``. The subset
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
 *     @buffer_format visBuffer structured
+*     @buffer_metadata visMetadata
 * @buffer out_buf The kotekan buffer which will be fed the subset of visibilities.
 *     @buffer_format visBuffer structured
+*     @buffer_metadata visMetadata
 *
 * @conf  prod_subset_type      string. Type of product subset to perform.
 * @conf  num_elements          int. The number of elements (i.e. inputs) in the
+*                              correlator data
+* @conf  num_ev                int. The number of eigenvectors to be stored
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
class prodSubset : public KotekanProcess {

public:
    /// Constructor. Loads config options. Defines subset of products.
    prodSubset(Config& config, const string& unique_name, bufferContainer& buffer_container);

    /// Primary loop: sorts products and passes them on to output buffer.
    void main_thread() override;

private:
    /// keeps track of the input dataset ID
    /// and gets new output dataset ID from manager
    ///
    dset_id_t change_dataset_state(dset_id_t ds_id, std::vector<prod_ctype>& prod_subset,
                                   std::vector<size_t>& prod_ind, size_t& subset_num_prod);

    /// Number of products in subset
    size_t subset_num_prod;

    /// Input buffer
    Buffer* in_buf;

    /// Output buffer to receive baseline subset visibilities
    Buffer* out_buf;

    /// Vector of indices for subset of products
    std::vector<size_t> prod_ind;

    /// Vector of subset of products
    std::vector<prod_ctype> prod_subset;

    // dataset IDs
    std::future<dset_id_t> future_output_dset_id;
};


/**
 * @brief Parse the product subseting section
 * @param config    Configuration handle.
 * @param base_path Path into YAML file to search from.
 * @return          Tuple containing a vector of the product inputs, and a
 *                  vector of the corresponding input labels.
 */
std::tuple<std::vector<size_t>, std::vector<prod_ctype>>
parse_prod_subset(Config& config, const std::string base_path);


#endif
