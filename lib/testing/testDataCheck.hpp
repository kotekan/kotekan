#ifndef TEST_DATA_CHECK_H
#define TEST_DATA_CHECK_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "errors.h"            // for TEST_PASSED
#include "kotekanLogging.hpp"  // for DEBUG, INFO, ERROR, FATAL_ERROR
#include "visUtil.hpp"         // for KOTEKAN_FLOAT16

#include "fmt.hpp"

#include <algorithm>   // for max
#include <assert.h>    // for assert
#include <cstdint>     // for int32_t, uint8_t, uint32_t
#include <exception>   // for exception
#include <functional>  // for bind
#include <limits>      // for numeric_limits
#include <math.h>      // for abs
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
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
    Buffer* first_buf;
    Buffer* second_buf;
    int num_frames_to_test;
    int max_num_errors_logged;
    double epsilon;
};

template<typename A_Type>
testDataCheck<A_Type>::testDataCheck(kotekan::Config& config, const std::string& unique_name,
                                     kotekan::bufferContainer& buffer_container) :
    kotekan::Stage(config, unique_name, buffer_container,
                   std::bind(&testDataCheck::main_thread, this)) {
    first_buf = get_buffer("first_buf");
    first_buf->register_consumer(unique_name);
    second_buf = get_buffer("second_buf");
    second_buf->register_consumer(unique_name);

    num_frames_to_test = config.get_default<int32_t>(unique_name, "num_frames_to_test", 0);
    max_num_errors_logged = config.get_default<int32_t>(unique_name, "max_num_errors_logged", 100);
    epsilon = config.get_default<double>(unique_name, "epsilon",
                                         std::numeric_limits<A_Type>::epsilon() * (A_Type)5.0);
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
        uint8_t* first_frame = first_buf->wait_for_full_frame(unique_name, first_buf_id);
        if (first_frame == nullptr)
            break;
        DEBUG("testDataCheck: Got the first buffer {:s}[{:d}]", first_buf->buffer_name,
              first_buf_id);
        uint8_t* second_frame = second_buf->wait_for_full_frame(unique_name, second_buf_id);
        if (second_frame == nullptr)
            break;
        DEBUG("testDataCheck: Got the second buffer {:s}[{:d}]", second_buf->buffer_name,
              second_buf_id);

        num_errors = 0;

        uint32_t num_elements = first_buf->frame_size / sizeof(A_Type);
        double abs_diff = 0.0;
        double rel_diff = 0.0;
        uint32_t num_nonzero = 0;

        bool use_almost_equal = !std::is_integral_v<A_Type> && (epsilon != 0.0);

        INFO(
            "Checking that the buffers {:s}[{:d}] and {:s}[{:d}] match, this could take a while...",
            first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);

        for (uint32_t i = 0; i < num_elements; ++i) {
            A_Type first_value = *((A_Type*)&(first_frame[i * sizeof(A_Type)]));
            A_Type second_value = *((A_Type*)&(second_frame[i * sizeof(A_Type)]));

            if (use_almost_equal) {
                double v1 = (double)first_value;
                double v2 = (double)second_value;
                abs_diff += std::abs(v1 - v2);
                double absum = (std::abs(v1) + std::abs(v2));
                if (absum != 0.0) {
                    rel_diff += std::abs(v1 - v2) / absum;
                    num_nonzero++;
                }

                if (!almost_equal(v1, v2, epsilon)) {
                    num_errors += 1;
                    if (num_errors < max_num_errors_logged) {
                        ERROR("{:s}[{:d}][{:d}] != {:s}[{:d}][{:d}]; values: ({:f}, {:f}), "
                              "epsilon: {:f}, "
                              "abs(x-y): {:f}, epsilon * abs(x+y): {:f}",
                              first_buf->buffer_name, first_buf_id, i, second_buf->buffer_name,
                              second_buf_id, i, v1, v2, epsilon,
                              std::abs((float)(first_value - second_value)),
                              epsilon * std::abs((float)(first_value + second_value)));
                    }
                }
            } else {
                if (first_value != second_value) {
                    num_errors += 1;
                    if (num_errors < max_num_errors_logged)
                        ERROR("{:s}[{:d}][{:d}] != {:s}[{:d}][{:d}]; values: ({:}, {:})",
                              first_buf->buffer_name, first_buf_id, i, second_buf->buffer_name,
                              second_buf_id, i, format_nice_string(first_value),
                              format_nice_string(second_value));
                }
            }
        }

        if (num_errors > 0) {
            INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained values that were equal.",
                 first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);
            if (use_almost_equal) {
                INFO("Compared {:d} elements.  Average absolute difference: {:g}.  Average "
                     "relative difference: {:g}",
                     num_elements, abs_diff / num_elements, rel_diff / num_elements);
                INFO("Nonzero elements: {:d}.  Average absolute difference: {:g}.  Average "
                     "relative difference: {:g}",
                     num_nonzero, abs_diff / std::max((uint32_t)1, num_nonzero),
                     rel_diff / std::max((uint32_t)1, num_nonzero));
            }
        } else {
            INFO("The buffers {:s}[{:d}] and {:s}[{:d}] contained values that were NOT equal!",
                 first_buf->buffer_name, first_buf_id, second_buf->buffer_name, second_buf_id);
            INFO("Test failed, exiting.");
            TEST_FAILED();
        }

        first_buf->mark_frame_empty(unique_name, first_buf_id);
        second_buf->mark_frame_empty(unique_name, second_buf_id);

        first_buf_id = (first_buf_id + 1) % first_buf->num_frames;
        second_buf_id = (second_buf_id + 1) % second_buf->num_frames;
        frames++;

        if (num_frames_to_test == frames) {
            if (num_errors == 0) {
                INFO("Test passed, exiting.");
                TEST_PASSED();
            }
        } // frames
    }
}

#endif
