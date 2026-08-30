#include "testDataCheckExpected.hpp"

#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE
#include "errors.h"               // for TEST_FAILED, TEST_PASSED
#include "kotekanLogging.hpp"     // for ERROR, FATAL_ERROR, INFO
#include "visUtil.hpp"            // for frameID, modulo
#include "waitingForAllTests.hpp" // for waiting_for_all_tests

#include "fmt.hpp" // for compile_string_to_view

#include <cmath>      // for fabs
#include <functional> // for bind, function

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataCheckExpected);

testDataCheckExpected::testDataCheckExpected(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&testDataCheckExpected::main_thread, this)),
    first_buf(get_buffer("first_buf")),
    second_buf(config.exists(unique_name, "second_buf") ? get_buffer("second_buf") : nullptr),
    expected(config.get<std::vector<float>>(unique_name, "expected")),
    epsilon(config.get_default<double>(unique_name, "epsilon", 1e-6)),
    num_frames_to_test(config.get<int32_t>(unique_name, "num_frames_to_test")),
    max_num_errors_logged(config.get_default<int32_t>(unique_name, "max_num_errors_logged", 100)),
    trigger_exit_on_pass(config.get_default<bool>(unique_name, "trigger_exit_on_pass", true)) {

    first_buf->register_consumer(unique_name);
    if (second_buf != nullptr) {
        second_buf->register_consumer(unique_name);
        if (second_buf->frame_size != first_buf->frame_size)
            FATAL_ERROR("first_buf frame size ({:d}) must equal second_buf frame size ({:d})",
                        first_buf->frame_size, second_buf->frame_size);
    }

    if (expected.empty())
        FATAL_ERROR("'expected' must hold at least one value");
    if (first_buf->frame_size % sizeof(float) != 0)
        FATAL_ERROR("first_buf frame size ({:d}) is not a whole number of floats",
                    first_buf->frame_size);

    check_component.assign(expected.size(), true);
    for (const int c :
         config.get_default<std::vector<int>>(unique_name, "skip_components", {})) {
        if (c < 0 || (size_t)c >= expected.size())
            FATAL_ERROR("skip_components entry {:d} outside expected list of {:d}", c,
                        expected.size());
        check_component[c] = false;
    }

    if (trigger_exit_on_pass)
        waiting_for_all_tests++;
}

void testDataCheckExpected::main_thread() {

    frameID first_id(first_buf);
    frameID second_id(second_buf != nullptr ? second_buf : first_buf);
    const size_t num_values = first_buf->frame_size / sizeof(float);
    int frames = 0;

    while (!stop_thread) {
        const float* first = (const float*)first_buf->wait_for_full_frame(unique_name, first_id);
        if (first == nullptr)
            break;
        const float* second = nullptr;
        if (second_buf != nullptr) {
            second = (const float*)second_buf->wait_for_full_frame(unique_name, second_id);
            if (second == nullptr)
                break;
        }

        int num_errors = 0;
        for (size_t i = 0; i < num_values; ++i) {
            const size_t c = i % expected.size();
            if (!check_component[c])
                continue;
            const double value = second != nullptr ? (double)first[i] - (double)second[i]
                                                   : (double)first[i];
            if (std::fabs(value - (double)expected[c]) > epsilon) {
                num_errors++;
                if (num_errors <= max_num_errors_logged)
                    ERROR("{:s}[{:d}][{:d}]: got {:g}, expected {:g} (component {:d})",
                          first_buf->buffer_name, first_id, i, value, expected[c], c);
            }
        }

        first_buf->mark_frame_empty(unique_name, first_id++);
        if (second_buf != nullptr)
            second_buf->mark_frame_empty(unique_name, second_id++);
        frames++;

        if (num_errors > 0) {
            ERROR("{:s}[{:d}]: {:d} of {:d} values did not match. Test failed, exiting.",
                  first_buf->buffer_name, first_id, num_errors, num_values);
            TEST_FAILED();
            break;
        }

        if (frames == num_frames_to_test) {
            if (trigger_exit_on_pass) {
                INFO("Test passed, exiting.");
                // Unregister to allow the pipeline to continue, unless I'm the last
                // consumer on this buffer.
                first_buf->unregister_consumer(unique_name, true);
                if (second_buf != nullptr)
                    second_buf->unregister_consumer(unique_name, true);
                if (--waiting_for_all_tests == 0) {
                    TEST_PASSED();
                }
            } else {
                INFO("Test passed.");
            }
            break;
        }
    }
}
