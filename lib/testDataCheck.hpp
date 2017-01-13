#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "buffers.h"
#include "KotekanProcess.hpp"


class testDataCheck : public KotekanProcess {
public:
    testDataCheck(Config &config,
                 struct Buffer &first_buf,
                 struct Buffer &second_buf);
    ~testDataCheck();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:
    struct Buffer &first_buf;
    struct Buffer &second_buf;
};

#endif