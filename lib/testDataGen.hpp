#ifndef TEST_DATA_GEN_H
#define TEST_DATA_GEN_H

#include "buffers.h"
#include "KotekanProcess.hpp"

class testDataGen : public KotekanProcess {
public:
    testDataGen(Config &config,
    			const string& unique_name,
                 struct Buffer &buf);
    ~testDataGen();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer &buf;
};

#endif