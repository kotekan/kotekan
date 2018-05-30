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
    void main_thread() override;
private:
    struct Buffer *buf;
    std::string type;
    int value;
    bool _pathfinder_test_mode;
    int samples_per_data_set;
    bool wait;
    std::string rest_mode;
    int num_frames;
    int stream_id;
};

#endif
