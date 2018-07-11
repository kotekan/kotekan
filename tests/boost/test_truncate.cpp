#define BOOST_TEST_MODULE "visTruncate"

#include <boost/test/included/unit_test.hpp>

#include "../../lib/utils/truncate.hpp"

BOOST_AUTO_TEST_CASE( _fast_pow)
{
    BOOST_CHECK_EQUAL(fast_pow(0), 1);
    BOOST_CHECK_EQUAL(fast_pow(1), 2);
    BOOST_CHECK_EQUAL(fast_pow(2), 4);
    BOOST_CHECK_EQUAL(fast_pow(5), 32);
    BOOST_CHECK_EQUAL(fast_pow(-1), 0.5);
    BOOST_CHECK_EQUAL(fast_pow(-2), 0.25);
    BOOST_CHECK_EQUAL(fast_pow(-4), 0.0625);
    BOOST_CHECK_EQUAL(fast_pow(64), std::pow(2.0, 64));
    BOOST_CHECK_EQUAL(fast_pow(-64), std::pow(2.0, -64));
    BOOST_CHECK_EQUAL(fast_pow(-127), std::pow(2.0, -127));
    BOOST_CHECK_EQUAL(fast_pow(INT8_MAX), std::pow(2.0, INT8_MAX));
    BOOST_CHECK_EQUAL(fast_pow(INT8_MIN),
            -1.0 * std::numeric_limits<float>::infinity());
}
