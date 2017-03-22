#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "buffers.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"

template <typename A_Type>
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

template <typename A_Type> testDataCheck<A_Type>::testDataCheck(Config& config,
                             Buffer& first_buf_,
                             Buffer& second_buf_) :
    KotekanProcess(config, std::bind(&testDataCheck::main_thread, this)),
    first_buf(first_buf_),
    second_buf(second_buf_) {

}

template <typename A_Type> testDataCheck<A_Type>::~testDataCheck() {

}

template <typename A_Type> void testDataCheck<A_Type>::apply_config(uint64_t fpga_seq) {

}

template <typename A_Type> void testDataCheck<A_Type>::main_thread() {

    int first_buf_id = 0;
    int second_buf_id = 0;
    int num_errors = 0;

    assert(first_buf.buffer_size == second_buf.buffer_size);

    for (;;) {

        // Get both full frames
        get_full_buffer_from_list(&first_buf, &first_buf_id, 1);
        INFO("testDataCheck: Got the first buffer %s[%d]", first_buf.buffer_name, first_buf_id);
//        get_full_buffer_from_list(&second_buf, &second_buf_id, 1);
//        INFO("testDataCheck: Got the second buffer %s[%d]", second_buf.buffer_name, second_buf_id);
        bool error = false;
        num_errors = 0;

        INFO("Checking that the buffers %s[%d] and %s[%d] match, this could take a while...",
                first_buf.buffer_name, first_buf_id,
                second_buf.buffer_name, second_buf_id);
        //hex_dump(16, (void*)first_buf.data[first_buf_id], 1024);
        //hex_dump(16, (void*)second_buf.data[second_buf_id], 1024);
        for (int i = 0; i < first_buf.buffer_size/sizeof(A_Type); ++i) {

            A_Type first_value = *((A_Type *)&(first_buf.data[first_buf_id][i*sizeof(A_Type)]));
            A_Type second_value;
            //second_value = *((A_Type *)&(second_buf.data[second_buf_id][i*sizeof(A_Type)]));
            if (i % 2 == 0) {
                second_value = 0.; //real part
            } else {
                second_value = 65536;//1605632.0;//9502720.0; //294912; //10256384; //163840;
            }

            if (first_value != second_value) {
                if (num_errors++ < 10000)
                ERROR("%s[%d][%d] != %s[%d][%d]; values: (%f, %f)",
                    first_buf.buffer_name, first_buf_id, i,
                    second_buf.buffer_name, second_buf_id, i,
                    (double)first_value, (double)second_value);
                error = true;
            }
        }

        if (!error)
            INFO("The buffers %s[%d] and %s[%d] are equal",
                    first_buf.buffer_name, first_buf_id,
                    second_buf.buffer_name, second_buf_id);

        mark_buffer_empty(&first_buf, first_buf_id);
//        mark_buffer_empty(&second_buf, second_buf_id);

        first_buf_id = (first_buf_id + 1) % first_buf.num_buffers;
        second_buf_id = (second_buf_id +1) % second_buf.num_buffers;

    }
}


#endif