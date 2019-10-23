#ifndef TEST_DATA_GEN_FLOAT_H
#define TEST_DATA_GEN_FLOAT_H

#include "Stage.hpp"
#include "buffer.h"

// Type: one of "random", "const"
// Value: the value of the constant
class testDataGenFloat : public kotekan::Stage {
public:
    testDataGenFloat(kotekan::Config& config, const string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~testDataGenFloat();
    void main_thread() override;

private:
    struct Buffer* buf;
    std::string type;
    float value;
    uint32_t _samples_per_data_set;
    bool _pathfinder_test_mode;
    uint32_t _first_frame_index;
};

#endif
