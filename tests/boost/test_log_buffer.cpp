#define BOOST_TEST_MODULE "test_log_buffer"

#include "errors.h"           // for MAX_LOG_MSG_LEN
#include "kotekanLogging.hpp" // for INFO_NON_OO
#include "logBuffer.h"        // for log_buffer_append, set_log_buffer, get_log_buffer

#include <boost/test/included/unit_test.hpp>
#include <string>   // for string, to_string
#include <syslog.h> // for LOG_INFO

using kotekan::get_log_buffer;
using kotekan::set_log_buffer;

BOOST_AUTO_TEST_CASE(off_keeps_nothing) {
    set_log_buffer(false);
    log_buffer_append(LOG_INFO, "", "dropped");
    BOOST_CHECK(get_log_buffer().empty());
}

BOOST_AUTO_TEST_CASE(keeps_the_last_lines) {
    set_log_buffer(true);
    for (int i = 0; i < LOG_BUFFER_LINES + 5; ++i)
        log_buffer_append(LOG_INFO, "", std::to_string(i).c_str());
    auto lines = get_log_buffer();
    BOOST_REQUIRE_EQUAL(lines.size(), size_t(LOG_BUFFER_LINES));
    BOOST_CHECK_EQUAL(lines.front()["msg"].get<std::string>(), "5");
    BOOST_CHECK_EQUAL(lines.back()["msg"].get<std::string>(), std::to_string(LOG_BUFFER_LINES + 4));
    BOOST_CHECK_EQUAL(lines.front()["level"].get<std::string>(), "INFO");
    set_log_buffer(false);
}

BOOST_AUTO_TEST_CASE(joins_prefix_and_cuts_long_lines) {
    set_log_buffer(true);
    log_buffer_append(LOG_INFO, "p", "m");
    log_buffer_append(LOG_INFO, "", std::string(2000, 'x').c_str());
    auto lines = get_log_buffer();
    BOOST_REQUIRE_EQUAL(lines.size(), 2u);
    BOOST_CHECK_EQUAL(lines[0]["msg"].get<std::string>(), "p: m");
    BOOST_CHECK_EQUAL(lines[1]["msg"].get<std::string>().size(), size_t(MAX_LOG_MSG_LEN));
    set_log_buffer(false);
}

BOOST_AUTO_TEST_CASE(records_logging_macros) {
    set_log_buffer(true);
    INFO_NON_OO("from the macro {:d}", 7);
    auto lines = get_log_buffer();
    BOOST_REQUIRE_EQUAL(lines.size(), 1u);
    BOOST_CHECK_EQUAL(lines[0]["msg"].get<std::string>(), "from the macro 7");
    set_log_buffer(false);
}

BOOST_AUTO_TEST_CASE(set_drops_kept_lines) {
    set_log_buffer(true);
    log_buffer_append(LOG_INFO, "", "old");
    set_log_buffer(true);
    BOOST_CHECK(get_log_buffer().empty());
    log_buffer_append(LOG_INFO, "", "new");
    set_log_buffer(false);
    BOOST_CHECK(get_log_buffer().empty());
}
