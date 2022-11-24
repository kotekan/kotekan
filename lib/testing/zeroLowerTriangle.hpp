#ifndef ZERO_LOWER_TRIANGLE_HPP
#define ZERO_LOWER_TRIANGLE_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h>    // for int32_t, uint32_t
#include <string>      // for string
#include <sys/types.h> // for uint

class zeroLowerTriangle : public kotekan::Stage {
public:
    zeroLowerTriangle(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~zeroLowerTriangle();
    void main_thread() override;

private:
    struct Buffer* input_buf;
    struct Buffer* output_buf;

    // Config options
    int32_t _num_local_freq;
    int32_t _num_elements;
    int32_t _samples_per_data_set;
    int32_t _sub_integration_ntime;
    std::string _data_format;
};

#endif
