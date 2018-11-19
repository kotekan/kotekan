#define BOOST_TEST_MODULE "test_visutils"

#include <boost/test/included/unit_test.hpp>

// the code to test:
#include "visUtil.hpp"

#include <stdio.h>

BOOST_AUTO_TEST_CASE( _divmod_pos )
{
    std::pair<int, int> answer = {1,0};
    std::pair<int, int> res = divmod_pos(1,1);
    BOOST_CHECK(res == answer);
}

BOOST_AUTO_TEST_CASE( _ts_to_double )
{
    timespec a = {10, 5};
    timespec b = {15, 2};

    BOOST_CHECK(ts_to_double(a) == 10.000000005);

    BOOST_CHECK(ts_to_double(b) == 15.000000002);
}

BOOST_AUTO_TEST_CASE( _timespec_substraction )
{
    timespec a = {10, 5};
    timespec b = {15, 2};
    BOOST_CHECK_THROW(subtract(a, b), std::runtime_error);

    a = {14, 99};
    b = {4, 88};
    timespec res = {10, 11};
    BOOST_CHECK(subtract(a, b) == res);

    a = {14, 1};
    b = {10, 2};
    res = {3, 999999999};
    BOOST_CHECK(subtract(a, b) == res);

    a = {1, 0};
    b = {2, 0};
    BOOST_CHECK_THROW(subtract(a, b), std::runtime_error);

    a = {1, 0};
    b = {1, 1};
    BOOST_CHECK_THROW(subtract(a, b), std::runtime_error);

    a = {2, 0};
    b = {1, 1};
    res = {0, 999999999};
    BOOST_CHECK(subtract(a, b) == res);

    a = {1, 0};
    b = {2, 1};
    BOOST_CHECK_THROW(subtract(a, b), std::runtime_error);

    a = {14, 99};
    b = {15, 101};
    BOOST_CHECK_THROW(subtract(a, b), std::runtime_error);
}