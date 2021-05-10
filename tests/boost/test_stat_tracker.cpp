#define BOOST_TEST_MODULE "test_stat_tracker"

#include <boost/test/included/unit_test.hpp>  // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...
#include <cmath>                              // for isnan

#include "visUtil.hpp"                        // for StatTracker

BOOST_AUTO_TEST_CASE(_stat_tracker_get_max) {
    StatTracker buf(3);

    BOOST_CHECK(isnan(buf.get_max()));
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_max(), 0.0);
    buf.add_sample(-2.5);
    BOOST_CHECK_EQUAL(buf.get_max(), 0.0);
    buf.add_sample(2.5);
    BOOST_CHECK_EQUAL(buf.get_max(), 2.5);

    // Overwrite the buffer with all 0.0 and see if the max can be updated.
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_max(), 2.5);
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_max(), 2.5);
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_max(), 0.0);
}

BOOST_AUTO_TEST_CASE(_stat_tracker_get_min) {
    StatTracker buf(3);

    BOOST_CHECK(isnan(buf.get_min()));
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_min(), 0.0);
    buf.add_sample(-2.5);
    BOOST_CHECK_EQUAL(buf.get_min(), -2.5);
    buf.add_sample(2.5);
    BOOST_CHECK_EQUAL(buf.get_min(), -2.5);

    // Overwrite the buffer with all 0.0 and see if the min can be updated.
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_min(), -2.5);
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_min(), 0.0);
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_min(), 0.0);
}

BOOST_AUTO_TEST_CASE(_stat_tracker_get_avg) {
    StatTracker buf(3);

    BOOST_CHECK(isnan(buf.get_avg()));
    buf.add_sample(0.0);
    BOOST_CHECK_EQUAL(buf.get_avg(), 0.0);
    buf.add_sample(1.0);
    BOOST_CHECK_EQUAL(buf.get_avg(), 0.5);
    buf.add_sample(2.0);
    BOOST_CHECK_EQUAL(buf.get_avg(), 1.0);

    // Overwrite the buffer to check the current average.
    buf.add_sample(3.0);
    BOOST_CHECK_EQUAL(buf.get_avg(), 2.0);
    buf.add_sample(4.0);
    BOOST_CHECK_EQUAL(buf.get_avg(), 3.0);
}

BOOST_AUTO_TEST_CASE(_stat_tracker_get_std_dev) {
    StatTracker buf(3);

    BOOST_CHECK(isnan(buf.get_std_dev()));
    buf.add_sample(0.0);
    BOOST_CHECK(isnan(buf.get_std_dev()));
    buf.add_sample(1.0);
    BOOST_CHECK_SMALL(buf.get_std_dev() - 0.707107, 0.000001);
    buf.add_sample(2.0);
    BOOST_CHECK_EQUAL(buf.get_std_dev(), 1.0);

    // Overwrite the buffer to check the current average.
    buf.add_sample(3.0);
    BOOST_CHECK_EQUAL(buf.get_std_dev(), 1.0);
    buf.add_sample(5.0);
    BOOST_CHECK_SMALL(buf.get_std_dev() - 1.527525, 0.000001);
}
