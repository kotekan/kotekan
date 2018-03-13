/*****************************************
@file
@brief Extract a subset of products from a visBuffer.
- prodSubset : public KotekanProcess
*****************************************/
#ifndef PROD_SUB
#define PROD_SUB

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"

/**
 * @class prodSubset
 * @brief ``KotekanProcess`` that consumes a full set of visibilities from a ``visBuffer``
 *        and passes on a subset of products to an output ``visBuffer``.
 * The subset extracted depends on the parameter 'prod_subset_type'. Here is a list of values
 * 'prod_subset_type' can take and the parameters they support: 
 * - 'autos': no extra arameters needed
 *   - The subset are all the auto-correlations.
 * - 'baseline': max_ew_baseline, max_ns_baseline
 *   - Selects a subset of products whose baseline length is smaller than the 
 *   maximum values given by config parameters in the EW and NS directions.
 * - 'input_list': input_list
 *   - The subset are all the correlations containing at leat one of the 
 *   inputs in the input_list.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer from which the visibilities are read, can be any size.
 *     @buffer_format visBuffer structured
+*     @buffer_metadata visMetadata
 * @buffer out_buf The kotekan buffer which will be fed the subset of visibilities.
 *     @buffer_format visBuffer structured
+*     @buffer_metadata visMetadata
 *
 * @conf  out_buf           string. Name of buffer to output subset to.
 * @conf  in_buf            string. Name of buffer to read from.
 * @conf  prod_subset_type  string. Type of product subset to perform.
 * @conf  num_elements      int. The number of elements (i.e. inputs) in the
+*                               correlator data
+* @conf  block_size        int. The block size of the packed data (read from "/")
 * @conf  num_prod          int. The number of products in the correlator data
 * @conf  subset_num_prod   int. The number of products in the subset data
 *                               (before subsetting)
+* @conf  num_eigenvectors  int. The number of eigenvectors to be stored
 * @conf  max_ew_baseline   int. The maximum baseline length along the EW direction to
 *                               include in subset (in units of the shortest EW baseline)
 * @conf  max_ns_baseline   int. The maximum baseline length along the NS direction to
 *                               include in subset (in units of the shortest NS baseline)
 * @conf  input_list        vector of int. The list of inputs to include.
 *
 * @author  Tristan Pinsonneault-Marotte and Mateus Fandino
 *
 */
class prodSubset : public KotekanProcess {

public:
    /// Constructor. Loads config options. Defines subset of products.
    prodSubset(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container);

    /// Not yet implemented, should update runtime parameters.
    void apply_config(uint64_t fpga_seq);

    /// Primary loop: sorts products and passes them on to output buffer.
    void main_thread();

private:
    /// Parameters saved from the config files
    size_t num_elements, num_eigenvectors, num_prod;

    /// Number of products in subset
    size_t subset_num_prod;

    /// Input buffer
    Buffer * in_buf;

    /// Output buffer to receive baseline subset visibilities
    Buffer* out_buf;

    /// Upper limits for baseline lengths in subset
    uint16_t xmax, ymax;

    /// Type of product subset to make
    std::string prod_subset_type;

    /// Vector of indices for subset of products
    std::vector<size_t> prod_ind;

    /// List of inputs whose correlations will be selected
    std::vector<int> input_list;
};


/**
 * @fn max_bl_condition
 *
 * Check if a correlation product corresponds to a baseline with length less
 * than or equal to specified maximum in the two grid directions.
 * Assumes the CHIME channel ordering.
 *
 * @param  prod     prod_ctype. The product pair to be checked.
 * @param  xmax     int. The maximum baseline length in the x direction.
 * @param  ymax     int. The maximum baseline length in the y direction.
 *
 * @return          bool. true if the product satisfies the condition.
 *
 */
inline bool max_bl_condition(prod_ctype prod, int xmax, int ymax) {

    // Figure out feed separations
    int x_sep = prod.input_a / 512 - prod.input_b / 512;
    int y_sep = prod.input_a % 256 - prod.input_b % 256;
    if (x_sep < 0) x_sep = - x_sep;
    if (y_sep < 0) y_sep = - y_sep;

    return (x_sep <= xmax) && (y_sep <= ymax);
}

/**
 * @overload max_bl_condition
 *
 * Accepts a visibility index (in the standard packing scheme) and the number
 * of elements in place of an explicit product pair.
 *
 * @param  vis_ind   int. Index of visibility in the standard UT packing.
 * @param  n         int. Total number of elements.
 *
 */
inline bool max_bl_condition(uint32_t vis_ind, int n, int xmax, int ymax) {

    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return max_bl_condition(prod, xmax, ymax);
}

/**
 * @fn have_inputs_condition
 *
 * Check if a correlation product contains at least one of the inputs
 * in the input_list parameter.
 * Assumes the CHIME channel ordering.
 *
 * @param  prod        prod_ctype. The product pair to be checked.
 * @param  input_list  vector of int. The list of inputs to include.
 *
 * @return          bool. true if the product satisfies the condition.
 *
 */
inline bool have_inputs_condition(prod_ctype prod, 
                                std::vector<int> input_list) {
   
    bool prod_in_list = false;
    for(auto ipt : input_list) {
        if ((prod.input_a==ipt) || (prod.input_b==ipt)) {
            prod_in_list = true;
            break;
        }
    }

    return prod_in_list;
}

/**
 * @overload have_inputs_condition
 *
 * Accepts a visibility index (in the standard packing scheme) and the number
 * of elements in place of an explicit product pair.
 *
 * @param  vis_ind   int. Index of visibility in the standard UT packing.
 * @param  n         int. Total number of elements.
 *
 */
inline bool have_inputs_condition(uint32_t vis_ind, int n, 
                                std::vector<int> input_list) {
    
    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return have_inputs_condition(prod, input_list);
}




inline bool only_inputs_condition(prod_ctype prod, 
                                std::vector<int> input_list) {
   
    bool ipta_in_list = false;
    bool iptb_in_list = false;
    for(auto ipt : input_list) {
        if (prod.input_a==ipt) {
            ipta_in_list = true;
        }
        if (prod.input_b==ipt) {
            iptb_in_list = true;
        }
    }

    return (ipta_in_list && iptb_in_list);
}


inline bool only_inputs_condition(uint32_t vis_ind, int n, 
                                std::vector<int> input_list) {
    
    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return only_inputs_condition(prod, input_list);
}


/**
 * @brief Parse the product subseting section
 * @param config    Configuration handle.
 * @param base_path Path into YAML file to search from.
 * @return          Tuple containing a vector of the product inputs, and a
 *                  vector of the corresponding input labels.
 */
std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> parse_prod_subset(Config& config, const std::string base_path);



#endif

