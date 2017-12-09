#include "visUtil.hpp"


// Copy the visibility triangle out of the buffer of data, allowing for a
// possible reordering of the inputs
void copy_vis_triangle(
    const int32_t * buf, const std::vector<uint32_t>& inputmap,
    size_t block, size_t N, complex_int * output
) {

    size_t pi = 0;
    uint32_t bi;

    if(*std::max_element(inputmap.begin(), inputmap.end()) >= N) {
        throw std::invalid_argument("Input map asks for elements out of range.");
    }

    for(auto i = inputmap.begin(); i != inputmap.end(); i++) {
        for(auto j = i; j != inputmap.end(); j++) {
            bi = prod_index(*i, *j, block, N);

            // IMPORTANT: for some reason the buffers are packed as imaginary
            // *then* real. Here we need to read out the individual components.
            output[pi].r = buf[2 * bi + 1];
            output[pi].i = buf[2 * bi];
            pi++;
        }
    }
}


std::vector<complex_int> copy_vis_triangle(
    const int32_t * buf, const std::vector<uint32_t>& inputmap,
    size_t block, size_t N
) {

    size_t M = inputmap.size();
    std::vector<complex_int> output(M * (M + 1) / 2);

    copy_vis_triangle(buf, inputmap, block, N, output.data());

    return output;
}
