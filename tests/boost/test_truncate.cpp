#define BOOST_TEST_MODULE "test_truncate"

#include "truncate.hpp" // for fast_pow, bit_truncate_float, count_zeros

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...
#include <limits>                            // for numeric_limits
#include <stdint.h>                          // for INT8_MAX, INT32_MAX, INT32_MIN, INT8_MIN


BOOST_AUTO_TEST_CASE(_fast_pow) {
    // results for fast_pow(e < -126) are not defined
    BOOST_CHECK_EQUAL(fast_pow(0), 1);
    BOOST_CHECK_EQUAL(fast_pow(1), 2);
    BOOST_CHECK_EQUAL(fast_pow(2), 4);
    BOOST_CHECK_EQUAL(fast_pow(5), 32);
    BOOST_CHECK_EQUAL(fast_pow(-1), 0.5);
    BOOST_CHECK_EQUAL(fast_pow(-2), 0.25);
    BOOST_CHECK_EQUAL(fast_pow(-4), 0.0625);
    BOOST_CHECK_EQUAL(fast_pow(64), std::pow(2.0, 64));
    BOOST_CHECK_EQUAL(fast_pow(-64), std::pow(2.0, -64));
    BOOST_CHECK_EQUAL(std::numeric_limits<float>::min(), (float)(std::pow(2.0, -126)));
    BOOST_CHECK_EQUAL(fast_pow(-125), (float)std::pow(2.0, -125));
    BOOST_CHECK_EQUAL(fast_pow(-126), (float)std::pow(2.0, -126));
    BOOST_CHECK_EQUAL(fast_pow(INT8_MAX), std::pow(2.0, INT8_MAX));
    BOOST_CHECK_EQUAL(fast_pow(INT8_MIN), -1.0 * std::numeric_limits<float>::infinity());
}

BOOST_AUTO_TEST_CASE(_count_zeros) {
    // int32_t count_zeros(int32_t x)
    BOOST_CHECK_EQUAL(count_zeros(0), 32);
    BOOST_CHECK_EQUAL(count_zeros(1), 31);
    BOOST_CHECK_EQUAL(count_zeros(2), 30);
    BOOST_CHECK_EQUAL(count_zeros(3), 30);
    BOOST_CHECK_EQUAL(count_zeros(INT32_MAX), 1);
    BOOST_CHECK_EQUAL(count_zeros(INT32_MIN), 0);
}

BOOST_AUTO_TEST_CASE(_bit_truncate_float) {
    // float bit_truncate_float(float val, float err)
    BOOST_CHECK_SMALL(bit_truncate_float(0.11, 0.01) - 0.11, 0.01);
    BOOST_CHECK(bit_truncate_float(0.11, 0.01) < 0.11);

    BOOST_CHECK_SMALL(bit_truncate_float(0.11, 0.01) - 0.11, 0.01);
    BOOST_CHECK(bit_truncate_float(0.11, 0.01) < 0.11);

    BOOST_CHECK(bit_truncate_float(0.0, 0.0) == 0.0);
    BOOST_CHECK_SMALL(bit_truncate_float(0.11, 0.0) - 0.11, 0.0001);
    BOOST_CHECK_SMALL(bit_truncate_float(1.11, 0.0) - 1.11, 0.0001);
    BOOST_CHECK(bit_truncate_float(0.0, 0.1) == 0.0);

    BOOST_CHECK_SMALL(bit_truncate_float(-0.11, 0.0) + 0.11, 0.0001);
    BOOST_CHECK_SMALL(bit_truncate_float(-1.11, 0.0) + 1.11, 0.0001);

    BOOST_CHECK_EQUAL(bit_truncate_float(std::numeric_limits<float>::max(), 0.01),
                      std::numeric_limits<float>::max());
    BOOST_CHECK_EQUAL(bit_truncate_float(std::numeric_limits<float>::min(), 0.01),
                      std::numeric_limits<float>::min());
}
