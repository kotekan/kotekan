#include "testDataCheck.hpp"
#include "errors.h"
#include "util.h"

testDataCheck::testDataCheck(Config& config,
                             Buffer& first_buf_,
                             Buffer& second_buf_) :
    KotekanProcess(config, std::bind(&testDataCheck::main_thread, this)),
    first_buf(first_buf_),
    second_buf(second_buf_) {

}

testDataCheck::~testDataCheck() {

}

void testDataCheck::apply_config(uint64_t fpga_seq) {

}

void testDataCheck::main_thread() {

    int first_buf_id = 0;
    int second_buf_id = 0;
    int num_errors = 0;

    assert(first_buf.buffer_size == second_buf.buffer_size);

    for (;;) {

        // Get both full frames
        get_full_buffer_from_list(&first_buf, &first_buf_id, 1);
        INFO("testDataCheck: Got the first buffer %s", first_buf.buffer_name);
        get_full_buffer_from_list(&second_buf, &second_buf_id, 1);
        INFO("testDataCheck: Got the second buffer %s", second_buf.buffer_name);
        bool error = false;
        num_errors = 0;

        INFO("Checking that the buffers %s[%d] and %s[%d] match, this could take a while...",
                first_buf.buffer_name, first_buf_id,
                second_buf.buffer_name, second_buf_id);
        //hex_dump(16, (void*)first_buf.data[first_buf_id], 1024);
        //hex_dump(16, (void*)second_buf.data[second_buf_id], 1024);
        for (int i = 0; i < first_buf.buffer_size/4; ++i) {
            // This could be made much faster with higher bit depth checks
            int32_t first_value = *((int32_t *)&(first_buf.data[first_buf_id][i*4]));
            int32_t second_value;
            second_value = *((int32_t *)&(second_buf.data[second_buf_id][i*4]));
            //if (i % 2 == 0) {
            //    second_value = 0;
            //} else {
            //    second_value = 294912; //10256384; //163840;
            //}

            if (first_value != second_value) {
                if (num_errors++ < 10000)
                ERROR("%s[%d][%d] != %s[%d][%d]; values: (%d, %d)",
                    first_buf.buffer_name, first_buf_id, i,
                    second_buf.buffer_name, second_buf_id, i,
                    first_value, second_value);
                error = true;
            }
        }

        if (!error)
            INFO("The buffers %s[%d] and %s[%d] are equal",
                    first_buf.buffer_name, first_buf_id,
                    second_buf.buffer_name, second_buf_id);

        mark_buffer_empty(&first_buf, first_buf_id);
        mark_buffer_empty(&second_buf, second_buf_id);

        first_buf_id = (first_buf_id + 1) % first_buf.num_buffers;
        second_buf_id = (second_buf_id +1) % second_buf.num_buffers;

    }
}


