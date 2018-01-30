#ifndef FAKE_GPU_BUFFER_HPP
#define FAKE_GPU_BUFFER_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"

class fakeGpuBuffer : public KotekanProcess {
public:
    fakeGpuBuffer(Config& config,
                const string& unique_name,
                bufferContainer &buffer_container);
    ~fakeGpuBuffer();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *output_buf;

    int freq;
    float cadence;

    int32_t block_size;
    int32_t num_blocks;
};

#endif
