#include "testDataGen.hpp"
#include <random>
#include "errors.h"

testDataGen::testDataGen(Config& config, Buffer& buf_) :
    KotekanProcess(config, std::bind(&testDataGen::main_thread, this)),
    buf(buf_) {

}

testDataGen::~testDataGen() {

}

void testDataGen::apply_config(uint64_t fpga_seq) {

}

void testDataGen::main_thread() {

    int buf_id = 0;
    int data_id = 0;
    uint64_t seq_num = 0;

    for (int i = 0; i < 10; ++i) {
        wait_for_empty_buffer(&buf, buf_id);

        set_data_ID(&buf, buf_id, data_id);
        set_fpga_seq_num(&buf, buf_id, seq_num);
        // TODO This should be dynamic/config controlled.
        set_stream_ID(&buf, buf_id, 0);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for (int j = 0; j < buf.buffer_size; ++j) {
            buf.data[buf_id][j] = (unsigned char)dis(gen);
        }

        INFO("Generated a test data set in %s[%d]", buf.buffer_name, buf_id);

        mark_buffer_full(&buf, buf_id);

        buf_id = (buf_id + 1) % buf.num_buffers;
        data_id++;
    }

}

