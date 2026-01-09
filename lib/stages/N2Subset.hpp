/*****************************************
@file
@brief Extract a subset of products from a N2 Buffer.
- N2Subset : public kotekan::Stage
*****************************************/
#ifndef N2_SUB_HPP
#define N2_SUB_HPP

#include "Config.hpp"
#include "N2Util.hpp" // for prod_ctype
#include "Stage.hpp"  // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <map>      // for map
#include <stddef.h> // for size_t
#include <string>   // for string
#include <tuple>    // for tuple
#include <utility>  // for pair
#include <vector>   // for vector


/**
 * @class N2Subset
 * @brief ``kotekan::Stage`` that extracts a subset of the products.
 *
 * This task consumes a full set of visibilities from a ``N2Buffer`` and
 * passes on a subset of products to an output ``N2Buffer``. The subset
 * extracted depends on the parameter 'prod_subset_type'. Here is a list of
 * values 'prod_subset_type' can take and the parameters they support:
 * - 'all': no extra parameters needed
 *   - Default. All the products.
 * - 'autos': no extra parameters needed
 *   - The subset are all the auto-correlations.
 * - 'have_inputs': input_list
 *   - The subset are all the correlations containing at least one of the
 *   inputs in the input_list.
 * - 'only_inputs': input_list
 *   - The subset are all the correlations containing only inputs from the
 *   input_list.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer from which the visibilities are read, can be any size.
 *     @buffer_format N2Buffer with FullUpperTri layout
 *     @buffer_metadata N2Metadata
 * @buffer out_buf The kotekan buffer which will be fed the subset of visibilities.
 *     @buffer_format N2Buffer with Autocorrelations layout
 *     @buffer_metadata N2Metadata
 *
 * @conf  prod_subset_type      string. Type of product subset to perform.
 * @conf  num_elements          UInt. The number of elements (i.e. inputs) in the
 *                              correlator data
 * @conf  input_list            vector of int. The list of inputs to include.
 *
 * @author  Tristan Pinsonneault-Marotte and Mateus Fandino
 *
 */
class N2Subset : public kotekan::Stage {

public:
    /// Constructor. Loads config options. Defines subset of products.
    N2Subset(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    /// Primary loop: sorts products and passes them on to output buffer.
    void main_thread() override;

private:
    /// Input buffer
    Buffer* in_buf;

    /// Output buffer to receive baseline subset visibilities
    Buffer* out_buf;

    /// vector of the product inputs
    const std::tuple<std::vector<size_t>, std::vector<N2::prod_ctype>> _prod_subset_list;

    /// Number of products after subsetting
    const size_t _num_subset_prod;

    /// Vector of indices for subset of products (before subsetting on the products in data)
    const std::vector<size_t> _base_prod_ind;

    /// Vector of subset of products
    const std::vector<N2::prod_ctype> _base_prod_subset;
};


/**
 * @brief Parse the product subseting section
 * @param config    kotekan::Configuration handle.
 * @param base_path Path into YAML file to search from.
 * @return          Tuple containing a vector of the product inputs, and a
 *                  vector of the corresponding input labels.
 */
std::tuple<std::vector<size_t>, std::vector<N2::prod_ctype>>
parse_N2_subset(kotekan::Config& config, const std::string base_path);


#endif
