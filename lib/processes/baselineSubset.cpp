#include "baselineSubset.hpp"
#include "visUtil.hpp"

baselineSubset::baselineSubset(Config &config,
                               const string& unique_name,
                               bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&fakeVis::main_thread, this)) {

    // Fetch any simple configuration
    num_elements = config.get_int("/", "num_elements");
    block_size = config.get_int("/", "block_size");
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
    num_prod = config.get_int("/", "num_prod");

    // Get the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buffer, unique_name.c_str());

    // Get the output buffers
    // One will receive all baselines, the other a subset
    // Ensure num_prod in metadata is adjusted

    // Define criteria for baseline selection based on config parameters
    xmax = config.get_int(unique_name, "max_x_baseline")
    ymax = config.get_int(unique_name, "max_y_baseline")

}

void baselineSubset::apply_config(uint64_t fpga_seq) {

}

void baselineSubset::main_thread() {

    while (!stop_thread) {

        // Read frame from input buffer

        // Check criteria to select a subset
        for (uint32_t i; i < num_prod; i++) {
            if (max_bl_condition(i, num_elements, xmax, ymax)) {
                // Copy this element to subset buffer
            }
        }

        // Copy frame to full buffer
        //      TODO: Is there some way to just allow another process to look at this frame next?
        //            Could just omit marking as empty. But then how to ensure other process
        //            doesn't eat frame before this one gets to it?

    }
}

// TODO: Should this be an inline function?
bool max_bl_condition(uint32_t vis_ind, int n, int xmax, int ymax) {

    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return max_bl_condition(prod, xmax, ymax);
}

bool max_bl_condition(prod_ctype prod, int xmax, int ymax) {

    // Figure out feed separations
    int x_sep = prod.input_a / 512 - prod.input_b / 512;
    int y_sep = prod.input_a % 256 - prod.input_b % 256;
    if (x_sep < 0) x_sep = - x_sep;
    if (y_sep < 0) y_sep = - y_sep;

    return (x_sep <= xmax) && (y_sep <= ymax);
}
