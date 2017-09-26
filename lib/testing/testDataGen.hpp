#ifndef TEST_DATA_GEN_H
#define TEST_DATA_GEN_H

#include "buffer.h"
#include "KotekanProcess.hpp"

// Type: one of "random", "const"
// Value: the value of the constant
class testDataGen : public KotekanProcess {
public:
    testDataGen(Config& config, const string& unique_name,
                bufferContainer &buffer_container);
    ~testDataGen();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf;
    std::string type;
    int32_t value;
};

#endif