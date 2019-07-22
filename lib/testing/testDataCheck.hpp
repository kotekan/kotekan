#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <type_traits>
#include <unistd.h>

template<typename A_Type>
class testDataCheck : public kotekan::Stage {
public:
    testDataCheck(kotekan::Config& config, const string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~testDataCheck();
    void main_thread() override;

private:
    struct Buffer* first_buf;
    struct Buffer* second_buf;
    int num_frames_to_test;
    float rel_diff_threshold;
};

template<typename A_Type>
testDataCheck<A_Type>::testDataCheck(kotekan::Config& config, const string& unique_name,
                                     kotekan::bufferContainer& buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container,
                   std::bind(&testDataCheck::main_thread, this)) {
    first_buf = get_buffer("first_buf");
    register_consumer(first_buf, unique_name.c_str());
    second_buf = get_buffer("second_buf");
    register_consumer(second_buf, unique_name.c_str());
    
    num_frames_to_test = config.get_default<int32_t>(unique_name, "num_frames_to_test", 0);
    rel_diff_threshold = config.get_default<float>(unique_name, "rel_diff_threshold", 0.001);
}

template<typename A_Type>
testDataCheck<A_Type>::~testDataCheck() {}

template<typename A_Type>
void testDataCheck<A_Type>::main_thread() {

    int first_buf_id = 0, second_buf_id = 0, num_errors = 0, frames = 0;
    const float abs_diff_threshold = 0.f;

    assert(first_buf->frame_size == second_buf->frame_size);

    while (!stop_thread) {

        // Get both full frames
        uint8_t* first_frame = wait_for_full_frame(first_buf, unique_name.c_str(), first_buf_id);
        if (first_frame == NULL)
            break;
        DEBUG("testDataCheck: Got the first buffer %s[%d]", first_buf->buffer_name, first_buf_id);
        uint8_t* second_frame = wait_for_full_frame(second_buf, unique_name.c_str(), second_buf_id);
        if (second_frame == NULL)
            break;
        DEBUG("testDataCheck: Got the second buffer %s[%d]", second_buf->buffer_name,
              second_buf_id);
        bool error = false;
        num_errors = 0;

        INFO("Checking that the buffers %s[%d] and %s[%d] match, this could take a while...",
             first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);

        for (uint32_t i = 0; i < first_buf->frame_size / sizeof(A_Type); ++i) {
            A_Type first_value = *((A_Type*)&(first_frame[i * sizeof(A_Type)]));
            A_Type second_value = *((A_Type*)&(second_frame[i * sizeof(A_Type)]));

            if ((std::is_same<A_Type, float>::value)
                or (std::is_same<A_Type, unsigned char>::value)) {
                float diff =
                    ((double)first_value - (double)second_value) / (double)first_value * 100;
                float diff2 = (double)first_value - (double)second_value;
                float diff3 =
                    ((double)first_value - (double)second_value) / (double)second_value * 100;
                if (((abs(diff) > rel_diff_threshold) and (abs(diff2) != abs_diff_threshold)) or (abs(diff3) > rel_diff_threshold)) {
                    error = true;
                    num_errors += 1;
                    if (num_errors < 20) {
                      FATAL_ERROR("%s[%d][%d] != %s[%d][%d]; values: (%f, %f) diffs (%.1f %.1f %.1f)",
                          first_buf->buffer_name, first_buf_id, i, second_buf->buffer_name,
                          second_buf_id, i, (double)first_value, (double)second_value, diff,
                          diff2, diff3);
                    }
                }
            } else { // N2 numbers are int
                // INFO("Checking non float numbers-----------");
                if (first_value != second_value) {
                    if (num_errors++ < 10000)
                        ERROR("%s[%d][%d] != %s[%d][%d]; values: (%f, %f)", first_buf->buffer_name,
                              first_buf_id, i, second_buf->buffer_name, second_buf_id, i,
                              (double)first_value, (double)second_value);
                    error = true;
                }
            }
        }

        if (!error) {
            INFO("The buffers %s[%d] and %s[%d] are equal", first_buf->buffer_name, first_buf_id,
                 second_buf->buffer_name, second_buf_id);
        }

        mark_frame_empty(first_buf, unique_name.c_str(), first_buf_id);
        mark_frame_empty(second_buf, unique_name.c_str(), second_buf_id);

        first_buf_id = (first_buf_id + 1) % first_buf->num_frames;
        second_buf_id = (second_buf_id + 1) % second_buf->num_frames;
        frames++;

        if(num_frames_to_test == frames) exit_kotekan(ReturnCode::CLEAN_EXIT);
    }
}

#endif
