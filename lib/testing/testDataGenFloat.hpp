#ifndef TEST_DATA_GEN_FLOAT_H
#define TEST_DATA_GEN_FLOAT_H

#include "KotekanProcess.hpp"
#include "buffer.h"

// Type: one of "random", "const"
// Value: the value of the constant
class testDataGenFloat : public kotekan::KotekanProcess {
public:
    testDataGenFloat(kotekan::Config& config, const string& unique_name,
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
