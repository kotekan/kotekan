#ifndef TEST_DATA_GEN_QUAD_H
#define TEST_DATA_GEN_QUAD_H

#include "buffer.h"
#include "KotekanProcess.hpp"

// Type: one of "random", "const"
// Value: the value of the constant
class testDataGenQuad : public KotekanProcess {
public:
    testDataGenQuad(Config& config, const string& unique_name,
                bufferContainer &buffer_container);
    ~testDataGenQuad();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer *buf[4];
    std::string type;
    vector<int32_t> value;
//    int32_t value;
};

#endif