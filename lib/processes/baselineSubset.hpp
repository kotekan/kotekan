#ifndef BASELINE_SUB
#define BASELINE_SUB

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"

class baselineSubset : public KotekanProcess {

public:
    baselineSubset(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:
    /// Parameters saved from the config files
    size_t num_elements, num_eigenvectors, num_prod;

    /// Input buffer
    Buffer * in_buf;

    /// Output buffer to receive baseline subset visibilities
    Buffer* out_buf;

    /// Upper limits for baseline lengths that will be passed to subset
    uint16_t xmax, ymax;

    /// Vector of indices for subset of products
    std::vector<size_t> prod_ind;

};


inline bool max_bl_condition(prod_ctype prod, int xmax, int ymax) {

    // Figure out feed separations
    int x_sep = prod.input_a / 512 - prod.input_b / 512;
    int y_sep = prod.input_a % 256 - prod.input_b % 256;
    if (x_sep < 0) x_sep = - x_sep;
    if (y_sep < 0) y_sep = - y_sep;

    return (x_sep <= xmax) && (y_sep <= ymax);
}

inline bool max_bl_condition(uint32_t vis_ind, int n, int xmax, int ymax) {

    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return max_bl_condition(prod, xmax, ymax);
}

#endif
