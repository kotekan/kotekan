#ifndef TIME_DOWNSAMP_HPP
#define TIME_DOWNSAMP_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"

class timeDownsamp : public KotekanProcess {

public:

    // Default constructor
    timeDownsamp(Config &config,
                const string& unique_name,
                bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    // Main loop for the process
    void main_thread();

private:

    // Parameters saved from the config files
    size_t num_elements, num_eigenvectors, block_size;
    size_t nprod;

    // Number of samples to combine
    int nsamp;

    // Buffers
    Buffer * in_buf;
    Buffer * out_buf;

};

#endif
