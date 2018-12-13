#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"
#include <unistd.h>
#include <type_traits>

template <typename A_Type>
class testDataCheck : public KotekanProcess {
public:
    testDataCheck(Config &config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~testDataCheck();
    void main_thread() override;
private:
    struct Buffer *first_buf;
    struct Buffer *second_buf;
};

template <typename A_Type> testDataCheck<A_Type>::testDataCheck(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&testDataCheck::main_thread, this)) {
    first_buf = get_buffer("first_buf");
    register_consumer(first_buf, unique_name.c_str());
    second_buf = get_buffer("second_buf");
    register_consumer(second_buf, unique_name.c_str());
}

template <typename A_Type> testDataCheck<A_Type>::~testDataCheck() {
}

template <typename A_Type> void testDataCheck<A_Type>::main_thread() {

    int first_buf_id = 0;
    int second_buf_id = 0;
    int num_errors = 0;

    assert(first_buf->frame_size == second_buf->frame_size);

    while(!stop_thread) {

        // Get both full frames
        uint8_t * first_frame = wait_for_full_frame(first_buf, unique_name.c_str(), first_buf_id);
        if (first_frame == NULL) break;
        DEBUG("testDataCheck: Got the first buffer %s[%d]", first_buf->buffer_name, first_buf_id);
        uint8_t * second_frame = wait_for_full_frame(second_buf, unique_name.c_str(), second_buf_id);
        if (second_frame == NULL) break;
        DEBUG("testDataCheck: Got the second buffer %s[%d]", second_buf->buffer_name, second_buf_id);
        bool error = false;
        num_errors = 0;

        INFO("Checking that the buffers %s[%d] and %s[%d] match, this could take a while...",
                first_buf->buffer_name, first_buf_id,
                second_buf->buffer_name, second_buf_id);

        for (uint32_t i = 0; i < first_buf->frame_size/sizeof(A_Type); ++i) {
            A_Type first_value = *((A_Type *)&(first_frame[i*sizeof(A_Type)]));
            A_Type second_value = *((A_Type *)&(second_frame[i*sizeof(A_Type)]));

            if ((std::is_same<A_Type, float>::value) or (std::is_same<A_Type, unsigned char>::value)) { 
                float diff = ((double)first_value - (double)second_value)/(double)first_value*100;
                float diff2 = (double)first_value - (double)second_value;
                float diff3 = ((double)first_value - (double)second_value)/(double)second_value*100;
                if (((abs(diff) > 0.001) and (abs(diff2) != 0.0)) or (abs(diff3) > 0.001)) {
                    error = true;
                    num_errors += 1;
                    if (num_errors <20 ) {
                        INFO("%s[%d][%d] != %s[%d][%d]; values: (%f, %f) diffs (%.1f %.1f %.1f)",
                         first_buf->buffer_name, first_buf_id, i,
                         second_buf->buffer_name, second_buf_id, i,
                         (double)first_value, (double)second_value, diff, diff2, diff3);
                    }
                }
            } 
	    else {  //N2 numbers are int
                //INFO("Checking non float numbers-----------");
                if (first_value != second_value) {
                    if (num_errors++ < 10000)
                        ERROR("%s[%d][%d] != %s[%d][%d]; values: (%f, %f)",
                          first_buf->buffer_name, first_buf_id, i,
                          second_buf->buffer_name, second_buf_id, i,
                          (double)first_value, (double)second_value);
                    error = true;
                }
            }
	    }

        if (!error) {
            INFO("The buffers %s[%d] and %s[%d] are equal",
                        first_buf->buffer_name, first_buf_id,
                        second_buf->buffer_name, second_buf_id);
        }

        mark_frame_empty(first_buf, unique_name.c_str(), first_buf_id);
        mark_frame_empty(second_buf, unique_name.c_str(), second_buf_id);

        first_buf_id = (first_buf_id + 1) % first_buf->num_frames;
        second_buf_id = (second_buf_id +1) % second_buf->num_frames;
    }
}

#endif
