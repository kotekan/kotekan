#ifndef GPU_SIMULATE_HPP
#define GPU_SIMULATE_HPP

#include "buffers.h"
#include "KotekanProcess.hpp"

class gpuSimulate : public KotekanProcess {
public:
    gpuSimulate(Config &config,
                struct Buffer &input_buf,
                struct Buffer &output_buf);
    ~gpuSimulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer &input_buf;
    struct Buffer &output_buf;

    uint32_t * host_block_map;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _num_blocks;
};

#endif