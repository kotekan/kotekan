#ifndef ACCUMULATE_HPP
#define ACCUMULATE_HPP

#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // IWYU pragma: keep

#include <stdint.h> // for int32_t
#include <string>   // for string

namespace kotekan {
class Config;
} // namespace kotekan

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
