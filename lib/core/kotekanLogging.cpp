#include "kotekanLogging.hpp"

#include "errors.h" // for __enable_syslog, __err_msg, __max_log_msg_len

#include "fmt.hpp" // for basic_string_view, print, vformat, basic_format_context, format_args

#include <stdexcept>   // for runtime_error
#include <stdio.h>     // for stderr
#include <strings.h>   // for strcasecmp
#include <type_traits> // for __underlying_type_impl<>::type, underlying_type

namespace kotekan {

kotekanLogging::kotekanLogging() {
    _member_log_level = 3;
    __log_prefix = "";
}

void kotekanLogging::vinternal_logging(int type, fmt::basic_string_view<char> log_prefix,
                                       const fmt::basic_string_view<char> format,
                                       fmt::format_args args) {
    std::string log_msg = fmt::vformat(format, args);
    if (log_prefix != "") {
        if (__enable_syslog == 1) {
            syslog(type, "%s: %s\n", log_prefix.data(), log_msg.data());
        } else {
            fmt::print(stderr, fmt("{:s}: {:s}\n"), log_prefix, log_msg);
        }
    } else {
        if (__enable_syslog == 1) {
            syslog(type, "%s\n", log_msg.data());
        } else {
            fmt::print(stderr, fmt("{:s}\n"), log_msg);
        }
    }
}

void kotekanLogging::set_log_level(const logLevel& log_level) {
    _member_log_level = static_cast<std::underlying_type<logLevel>::type>(log_level);
}

void kotekanLogging::set_log_prefix(const std::string& log_prefix) {
    __log_prefix = log_prefix;
}

void kotekanLogging::set_log_level(const std::string& s_log_level) {

    logLevel log_level;

    if (strcasecmp(s_log_level.c_str(), "off") == 0) {
        log_level = logLevel::OFF;
    } else if (strcasecmp(s_log_level.c_str(), "error") == 0) {
        log_level = logLevel::ERROR;
    } else if (strcasecmp(s_log_level.c_str(), "warn") == 0) {
        log_level = logLevel::WARN;
    } else if (strcasecmp(s_log_level.c_str(), "info") == 0) {
        log_level = logLevel::INFO;
    } else if (strcasecmp(s_log_level.c_str(), "debug") == 0) {
        log_level = logLevel::DEBUG;
    } else if (strcasecmp(s_log_level.c_str(), "debug2") == 0) {
        log_level = logLevel::DEBUG2;
    } else {
        throw std::runtime_error(fmt::format(fmt("The value given for log_level: '{:s}' is not "
                                                 "valid! (It should be one of 'off', 'error', "
                                                 "'warn', 'info', 'debug', 'debug2')"),
                                             s_log_level));
    }

    set_log_level(log_level);
}

void kotekanLogging::vset_error_message(const fmt::basic_string_view<char> format,
                                        fmt::format_args args) {
    fmt::format_to_n(__err_msg, __max_log_msg_len, fmt::vformat(format, args));
}

} // namespace kotekan
