#ifndef GPU_SIMULATE_HPP
#define GPU_SIMULATE_HPP

#include "Stage.hpp"
#include "buffer.h"

class gpuSimulate : public kotekan::Stage {
public:
    gpuSimulate(kotekan::Config& config, const string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~gpuSimulate();
    void main_thread() override;

private:
    int dot4b(uint a, uint b);

    struct Buffer* input_buf;
    struct Buffer* output_buf;

    uint32_t* host_block_map;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _num_blocks;
    int32_t _block_size;
    string _data_format;
};

#endif
