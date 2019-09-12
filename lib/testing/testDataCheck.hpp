#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "Stage.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <cmath>
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
    double epsilon;
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
    epsilon = config.get_default<double>(unique_name, "epsilon",
                                         std::numeric_limits<A_Type>::epsilon() * 5);
}

template<typename A_Type>
testDataCheck<A_Type>::~testDataCheck() {}

template<typename A_Type>
typename std::enable_if<!std::numeric_limits<A_Type>::is_integer, bool>::type
almost_equal(A_Type x, A_Type y, double epsilon) {
    // the machine epsilon has to be scaled to the magnitude of the values
    return std::abs(x - y) <= epsilon * std::abs(x + y)
           // unless the result is subnormal
           || std::abs(x - y) < std::numeric_limits<A_Type>::min();
}

template<typename A_Type>
void testDataCheck<A_Type>::main_thread() {

    int first_buf_id = 0, second_buf_id = 0, num_errors = 0, frames = 0;

    assert(first_buf->frame_size == second_buf->frame_size);

    while (!stop_thread) {

        // Get both full frames
        uint8_t* first_frame = wait_for_full_frame(first_buf, unique_name.c_str(), first_buf_id);
        if (first_frame == NULL)
            break;
        DEBUG("testDataCheck: Got the first buffer {:s}[{:d}]", first_buf->buffer_name,
              first_buf_id);
        uint8_t* second_frame = wait_for_full_frame(second_buf, unique_name.c_str(), second_buf_id);
        if (second_frame == NULL)
            break;
        DEBUG("testDataCheck: Got the second buffer {:s}[{:d}]", second_buf->buffer_name,
              second_buf_id);
        bool error = false;
        num_errors = 0;

        INFO(
            "Checking that the buffers {:s}[{:d}] and {:s}[{:d}] match, this could take a while...",
            first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);

        for (uint32_t i = 0; i < first_buf->frame_size / sizeof(A_Type); ++i) {
            A_Type first_value = *((A_Type*)&(first_frame[i * sizeof(A_Type)]));
            A_Type second_value = *((A_Type*)&(second_frame[i * sizeof(A_Type)]));

            if ((std::is_same<A_Type, float>::value)
                or (std::is_same<A_Type, unsigned char>::value)) {
                if (!almost_equal((double)first_value, (double)second_value, epsilon)) {
                    error = true;
                    num_errors += 1;
                    if (num_errors < 20) {
                        FATAL_ERROR("{:s}[{:d}][{:d}] != {:s}[{:d}][{:d}]; values: ({:f}, {:f}), "
                                    "epsilon: {:f}, "
                                    "abs(x-y): {:f}, epsilon * abs(x+y): {:f}",
                                    first_buf->buffer_name, first_buf_id, i,
                                    second_buf->buffer_name, second_buf_id, i, (double)first_value,
                                    (double)second_value, epsilon,
                                    std::abs(first_value - second_value),
                                    epsilon * std::abs(first_value + second_value));
                    }
                }
            } else { // N2 numbers are int
                // INFO("Checking non float numbers-----------");
                if (first_value != second_value) {
                    if (num_errors++ < 20)
                        ERROR("{:s}[{:d}][{:d}] != {:s}[{:d}][{:d}]; values: ({:f}, {:f})",
                              first_buf->buffer_name, first_buf_id, i, second_buf->buffer_name,
                              second_buf_id, i, (double)first_value, (double)second_value);
                    error = true;
                }
            }
        }

        if (!error) {
            INFO("The buffers {:s}[{:d}] and {:s}[{:d}] are equal", first_buf->buffer_name,
                 first_buf_id, second_buf->buffer_name, second_buf_id);
        }

        mark_frame_empty(first_buf, unique_name.c_str(), first_buf_id);
        mark_frame_empty(second_buf, unique_name.c_str(), second_buf_id);

        first_buf_id = (first_buf_id + 1) % first_buf->num_frames;
        second_buf_id = (second_buf_id + 1) % second_buf->num_frames;
        frames++;

        if (num_frames_to_test == frames)
            TEST_PASSED();
    }
}

#endif
