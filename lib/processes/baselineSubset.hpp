/*****************************************
File Contents:
- baselineSubset : public KotekanProcess
*****************************************/
#ifndef BASELINE_SUB
#define BASELINE_SUB

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"

/**
 * @class baselineSubset
 * @brief ``KotekanProcess`` that consumes a full set of visibilities from a ``visBuffer``
 *        and passes on a subset of these to an output ``visBuffer``.
 *
 * This process selects a subset of Pathfinder-scale baselines from the full visibility
 * array and passes those on to an output buffer. The conditions that define the subset
 * are specified in the config as maximum baseline lengths in the EW and NS directions.
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
 * @conf  num_elements      int. The number of elements (i.e. inputs) in the
+*                               correlator data (read from "/")
+* @conf  block_size        int. The block size of the packed data (read from "/")
 * @conf  num_prod          int. The number of products in the correlator data
 *                               (before subsetting) (read from "/")
+* @conf  num_eigenvectors  int. The number of eigenvectors to be stored
 * @conf  max_ew_baseline   int. The maximum baseline length along the EW direction to
 *                               include in subset (in units of the shortest EW baseline)
 * @conf  max_ns_baseline   int. The maximum baseline length along the NS direction to
 *                               include in subset (in units of the shortest NS baseline)
 *
 * @author  Tristan Pinsonneault-Marotte
 *
 */
class baselineSubset : public KotekanProcess {

public:
    /// Constructor. Loads config options. Defines subset of products.
    baselineSubset(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container);

    /// Not yet implemented, should update runtime parameters.
    void apply_config(uint64_t fpga_seq);

    /// Primary loop: sorts products and passes them on to output buffer.
    void main_thread();

private:
    /// Parameters saved from the config files
    size_t num_elements, num_eigenvectors, num_prod;

    /// Input buffer
    Buffer * in_buf;

    /// Output buffer to receive baseline subset visibilities
    Buffer* out_buf;

    /// Upper limits for baseline lengths in subset
    uint16_t xmax, ymax;

    /// Vector of indices for subset of products
    std::vector<size_t> prod_ind;

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

#endif
