//#define BOOST_TEST_MODULE "test_fmt"
//#include <boost/test/included/unit_test.hpp>

#include <chrono>     // for duration, operator-, seconds, operator/, operator>, tim...
#include "fmt.hpp" // for format
#include "kotekanLogging.hpp"    // for DEBUG, INFO, ERROR, FATAL_ERROR, WARN
#include "errors.h"

//BOOST_AUTO_TEST_CASE(test1) {
int main() {
    _global_log_level = 4;
    __enable_syslog = 0;
    
    std::chrono::time_point<std::chrono::steady_clock> period_start = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> diff = now - period_start;
    INFO_NON_OO("duration {}", diff);
    INFO_NON_OO("duration {:.3f}", diff.count());
    // FAILS with a fmt error
    INFO_NON_OO("duration {:.3f}", diff);
}
