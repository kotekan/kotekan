#define BOOST_TEST_MODULE "test_<name>"

#include <boost/test/included/unit_<test>.hpp>

// include the code you want to test:
#include "<name>.hpp"

// Split your tests into test cases
BOOST_AUTO_TEST_CASE( _some_useful_test_case_name )
{
    BOOST_CHECK_EQUAL(function(1), 1);
    BOOST_CHECK_EQUAL(function(6), 42);
    BOOST_CHECK_EQUAL(function(26), 4861946401452);
}
