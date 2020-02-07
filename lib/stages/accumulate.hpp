#ifndef ACCUMULATE_HPP
#define ACCUMULATE_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for int32_t
#include <string>   // for string


class accumulate : public kotekan::Stage {
public:
    accumulate(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);
    ~accumulate();
    void main_thread() override;

private:
    struct Buffer* in_buf;
    struct Buffer* out_buf;

    int32_t _samples_per_data_set;
    int32_t _num_gpu_frames;
};

#endif
