#ifndef TEST_DATA_GEN_FLOAT_H
#define TEST_DATA_GEN_FLOAT_H

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for datatype_t
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string

// Type: one of "random", "const"
// Value: the value of the constant
class testDataGenFloat : public kotekan::Stage {
public:
    testDataGenFloat(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~testDataGenFloat();
    void main_thread() override;

private:
    Buffer* buf;
    std::string type;
    int seed;
    float value;
    uint32_t _samples_per_data_set;
    bool _pathfinder_test_mode;
    uint32_t _first_frame_index;
    bool _gen_all_const_data;
    float _rand_min;
    float _rand_max;

    std::string _name;
    std::vector<int> _array_shape;
    std::vector<std::string> _dim_name;
    std::vector<ptrdiff_t> _dim_scaling;
    kotekan::DataType _value_type;
    size_t _num_freq_in_frame;
    std::vector<uint32_t> _manual_freq_ids;
    int _meta_time_downsample_factor;
};

#endif
