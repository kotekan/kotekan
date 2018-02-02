#ifndef EIGENVIS_HPP
#define EIGENVIS_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"

class eigenVis : public KotekanProcess {
public:
    eigenVis(Config& config,
                const string& unique_name,
                bufferContainer &buffer_container);
    ~eigenVis();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *in_buf;
    struct Buffer *out_buf;

    int32_t num_eigenvectors;
};

#endif
