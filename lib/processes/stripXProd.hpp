#ifndef STRIP_XPROD_HPP
#define STRIP_XPROD_HPP

#include <unistd.h>
#include "buffer.h"
#include "KotekanProcess.hpp"


class stripXProd : public KotekanProcess {

public:

    /// Default constructor
    stripXProd(Config &config,
               const string& unique_name,
               bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    /// Main loop for the process
    void main_thread();

private:
// TODO: delete
//    // List of frequencies for the subset
//    std::vector<uint32_t> subset_list;

    /// Output buffer with subset of frequencies
    Buffer * out_buf;
    /// Input buffer with all frequencies
    Buffer * in_buf;
};



#endif

