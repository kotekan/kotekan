#ifndef GPU_SIMULATE_N2K_HPP
#define GPU_SIMULATE_N2K_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for int32_t
#include <string>   // for string

class gpuSimulateN2k : public kotekan::Stage {
public:
    gpuSimulateN2k(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~gpuSimulateN2k();
    void main_thread() override;

private:
    struct Buffer* input_buf;
    struct Buffer* output_buf;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _sub_integration_ntime;
};

#endif
