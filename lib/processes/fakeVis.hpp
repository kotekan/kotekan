#ifndef FAKE_VIS
#define FAKE_VIS

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"

// Generated fake visibility buffers for downstream task development
class fakeVis : public KotekanProcess {

public:
    fakeVis(Config &config,
            const string& unique_name,
            bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:
    // Parameters saved from the config files
    size_t num_elements, num_eigenvectors, block_size;

    // Output buffer
    // TODO: Should have a single output buffer
    //       frames per freq per time
    Buffer * output_buffer;

    // List of frequencies for every buffer
    std::vector<uint16_t> freq;

};

#endif
