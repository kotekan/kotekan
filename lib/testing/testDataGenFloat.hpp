#ifndef TEST_DATA_GEN_FLOAT_H
#define TEST_DATA_GEN_FLOAT_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string

// Type: one of "random", "const"
// Value: the value of the constant
class testDataGenFloat : public kotekan::Stage {
public:
    testDataGenFloat(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~testDataGenFloat();
    void main_thread() override;

private:
    struct Buffer* buf;
    std::string type;
    int seed;
    float value;
    uint32_t _samples_per_data_set;
    bool _pathfinder_test_mode;
    uint32_t _first_frame_index;
    bool _gen_all_const_data;
    float _rand_min;
    float _rand_max;
};

#endif
