#define BOOST_TEST_MODULE "test_fmt"
#include "errors.h"
#include "kotekanLogging.hpp" // for DEBUG, INFO, ERROR, FATAL_ERROR, WARN

#include "fmt.hpp" // for format

#include <boost/test/included/unit_test.hpp>
#include <chrono> // for duration, operator-, seconds, operator/, operator>, tim...
#include <thread>

BOOST_AUTO_TEST_CASE(test_output_macros) {
    _global_log_level = 4;
    __enable_syslog = 0;

    using namespace std::chrono_literals;

    std::chrono::time_point<std::chrono::steady_clock> period_start =
        std::chrono::steady_clock::now();
    std::this_thread::sleep_for(1000ms);
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> diff = now - period_start;
    INFO_NON_OO("duration {}", diff);
    INFO_NON_OO("duration {:.3f}", diff.count());

    // FAILS with a fmt error
    // Catch fmt error
    BOOST_CHECK_THROW(INFO_NON_OO("duration {:.3f}", diff), fmt::v10::format_error);

    // Test other macros
    WARN_NON_OO("duration {:.3f}", diff.count());
    DEBUG_NON_OO("duration {:.3f}", diff.count());
    DEBUG2_NON_OO("duration {:.3f}", diff.count());
}

BOOST_AUTO_TEST_CASE(test_error_message_keeps_braces) {
    // The stored message is the formatted text verbatim; a brace in an argument or an escaped
    // brace in the format must come through unchanged rather than be parsed again.
    kotekan::kotekanLogging::set_error_message(fmt("Failed to deserialize from {:s}: {:s}"),
                                               std::string("10.0.0.1"),
                                               std::string("expected '[', '{', or a literal"));
    BOOST_CHECK_EQUAL(std::string(get_error_message()),
                      "Failed to deserialize from 10.0.0.1: expected '[', '{', or a literal");

    kotekan::kotekanLogging::set_error_message(fmt("must be an object (e.g. {{enabled: true}})"));
    BOOST_CHECK_EQUAL(std::string(get_error_message()), "must be an object (e.g. {enabled: true})");

    // A message longer than the buffer is cut, still NUL-terminated.
    kotekan::kotekanLogging::set_error_message(fmt("{:s}"), std::string(2 * MAX_LOG_MSG_LEN, 'x'));
    BOOST_CHECK_EQUAL(std::string(get_error_message()), std::string(MAX_LOG_MSG_LEN - 1, 'x'));
}
