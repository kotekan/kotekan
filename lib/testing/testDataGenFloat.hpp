#ifndef TEST_DATA_GEN_FLOAT_H
#define TEST_DATA_GEN_FLOAT_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <string> // for string

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
    float value;
    bool _pathfinder_test_mode;
};

#endif
