#define BOOST_TEST_MODULE "test_visutils"

#include <boost/test/included/unit_test.hpp>

// the code to test:
#include "visUtil.hpp"

#include <chrono>

BOOST_AUTO_TEST_CASE(_divmod_pos) {
    std::pair<int, int> answer = {1, 0};
    std::pair<int, int> res = divmod_pos(1, 1);
    BOOST_CHECK(res == answer);
}

BOOST_AUTO_TEST_CASE(_ts_to_double) {
    timespec a = {10, 5};
    timespec b = {15, 2};

    BOOST_CHECK(ts_to_double(a) == 10.000000005);

    BOOST_CHECK(ts_to_double(b) == 15.000000002);
}

BOOST_AUTO_TEST_CASE(_timespec_addition) {
    __log_level = 5;
    __enable_syslog = 0;

    timespec a = {10, 5};
    timespec b = {10, 0};
    timespec expected = {20, 5};
    BOOST_CHECK(a + b == expected);

    timespec c = {0, 10};
    expected = {10, 15};
    BOOST_CHECK(a + c == expected);

    timespec d = {10, 10};
    expected = {20, 15};
    BOOST_CHECK(a + d == expected);
}

BOOST_AUTO_TEST_CASE(_timespec_subtraction) {
    __log_level = 5;
    __enable_syslog = 0;
    // positive durations
    timespec start = {10, 0};
    timespec end = {10, 5};
    std::chrono::nanoseconds res = difference(start, end);
    BOOST_CHECK(res == std::chrono::nanoseconds(5));

    start = {0, 10};
    res = difference(start, end);
    BOOST_CHECK(res
                == std::chrono::seconds(10) + std::chrono::nanoseconds(5)
                       - std::chrono::nanoseconds(10));

    start = {10, 5};
    res = difference(start, end);
    BOOST_CHECK(res == std::chrono::seconds(0));

    start = {9, 4};
    res = difference(start, end);
    BOOST_CHECK(res == std::chrono::seconds(1) + std::chrono::nanoseconds(1));

    // negative durations
    start = {10, 10};
    res = difference(start, end);
    BOOST_CHECK(res == std::chrono::nanoseconds(-5));

    start = {11, 10};
    res = difference(start, end);
    BOOST_CHECK(res == std::chrono::seconds(-1) + std::chrono::nanoseconds(-5));

    start = {100, 0};
    end = {10, 0};
    res = difference(start, end);
    BOOST_CHECK(res == std::chrono::seconds(-90));
}
