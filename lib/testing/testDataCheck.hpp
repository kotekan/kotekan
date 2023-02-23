#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "errors.h"            // for TEST_PASSED
#include "kotekanLogging.hpp"  // for DEBUG, INFO, ERROR, FATAL_ERROR

#include "fmt.hpp"

#include <assert.h>    // for assert
#include <cstdint>     // for int32_t, uint8_t, uint32_t
#include <exception>   // for exception
#include <functional>  // for bind
#include <limits>      // for numeric_limits
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdlib.h>    // for abs
#include <string>      // for string, allocator
#include <type_traits> // for is_same, enable_if
#include <vector>      // for vector


template<typename A_Type>
class testDataCheck : public kotekan::Stage {
public:
    testDataCheck(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~testDataCheck();
    void main_thread() override;

private:
    struct Buffer* first_buf;
    struct Buffer* second_buf;
    int num_frames_to_test;
    int max_num_errors;
    double epsilon;
};

template<typename A_Type>
testDataCheck<A_Type>::testDataCheck(kotekan::Config& config, const std::string& unique_name,
                                     kotekan::bufferContainer& buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container,
                   std::bind(&testDataCheck::main_thread, this)) {
    first_buf = get_buffer("first_buf");
    register_consumer(first_buf, unique_name.c_str());
    second_buf = get_buffer("second_buf");
    register_consumer(second_buf, unique_name.c_str());

    num_frames_to_test = config.get_default<int32_t>(unique_name, "num_frames_to_test", 0);
    max_num_errors = config.get_default<int32_t>(unique_name, "max_num_errors", 100);
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

static std::string format_nice_string(uint8_t x) {
    return fmt::format("{} = 0x{:x}", x, x);
}
/*static std::string format_nice_string(uint32_t x) {
    return fmt::format("{} = 0x{:x}", x, x);
    }*/
static std::string format_nice_string(int x) {
    return fmt::format("{} = 0x{:x}", x, x);
}
#if KOTEKAN_FLOAT16
static std::string format_nice_string(float16_t x) {
    return fmt::format("{}", (float)x);
}
#endif
static std::string format_nice_string(float x) {
    return fmt::format("{}", x);
}

template<typename A_Type>
void testDataCheck<A_Type>::main_thread() {

    int first_buf_id = 0, second_buf_id = 0, num_errors = 0, frames = 0;

    assert(first_buf->frame_size == second_buf->frame_size);

    while (!stop_thread) {

        // Get both full frames
        uint8_t* first_frame = wait_for_full_frame(first_buf, unique_name.c_str(), first_buf_id);
        if (first_frame == nullptr)
            break;
        DEBUG("testDataCheck: Got the first buffer {:s}[{:d}]", first_buf->buffer_name,
              first_buf_id);
        uint8_t* second_frame = wait_for_full_frame(second_buf, unique_name.c_str(), second_buf_id);
        if (second_frame == nullptr)
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

            if (std::is_same<A_Type, float>::value) {
                if (!almost_equal((double)first_value, (double)second_value, epsilon)) {
                    error = true;
                    num_errors += 1;
                    if (num_errors < max_num_errors) {
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
                    if (num_errors++ < max_num_errors)
                        ERROR("{:s}[{:d}][{:d}] != {:s}[{:d}][{:d}]; values: ({:}, {:})",
                              first_buf->buffer_name, first_buf_id, i, second_buf->buffer_name,
                              second_buf_id, i, format_nice_string(first_value),
                              format_nice_string(second_value));
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

        if (num_frames_to_test == frames) {
            INFO("Test passed, exiting!");
            TEST_PASSED();
        }
    }
}

#endif
