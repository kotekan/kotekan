#define BOOST_TEST_MODULE "test_timeUtil"

#include <boost/test/included/unit_test.hpp>
#include <inttypes.h>
#include <time.h>
#include "timeUtil.hpp"


void check_timespec_equal(const timespec &t1, const timespec &t2) {
    BOOST_CHECK_EQUAL(t1.tv_sec, t2.tv_sec);
    BOOST_CHECK_EQUAL(t1.tv_nsec, t2.tv_nsec);
}

BOOST_AUTO_TEST_CASE(_time_to_ut1) {

    timespec t_J2000 = {.tv_sec=946'727'935L, .tv_nsec=816'000'000L};
    timespec t_J2000_ut1 = {.tv_sec=2'451'545L * 86400L - 65,
                            .tv_nsec=816'000'000L};

    check_timespec_equal(get_UT1_from_time(t_J2000, 0.0), t_J2000_ut1);
}
