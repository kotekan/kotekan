#include "kotekanLogging.hpp"

#include <stdexcept>
#include <strings.h>

namespace kotekan {

kotekanLogging::kotekanLogging() {
    __log_level = 3;
    __log_prefix = "";
}

void kotekanLogging::internal_logging(int type, const char* format, ...) {
    char log_buf[__max_log_msg_len];
    if (__log_prefix != "") {
        va_list args;
        va_start(args, format);
        vsnprintf(log_buf, __max_log_msg_len, format, args);
        va_end(args);
        if (__enable_syslog == 1) {
            syslog(type, "%s: %s\n", __log_prefix.c_str(), log_buf);
        } else {
            fprintf(stderr, "%s: %s\n", __log_prefix.c_str(), log_buf);
        }
    } else {
        va_list args;
        va_start(args, format);
        if (__enable_syslog == 1) {
            (void)vsyslog(type, format, args);
        } else {
            (void)vsnprintf(log_buf, __max_log_msg_len, format, args);
            fprintf(stderr, "%s\n", log_buf);
        }
        va_end(args);
    }
}

void kotekanLogging::set_log_level(const logLevel& log_level) {
    __log_level = static_cast<std::underlying_type<logLevel>::type>(log_level);
}

void kotekanLogging::set_log_prefix(const string& log_prefix) {
    __log_prefix = log_prefix;
}

void kotekanLogging::set_log_level(const string& s_log_level) {

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
        throw std::runtime_error(
            "The value given for log_level: '" + s_log_level + "is not valid! "
            + "(It should be one of 'off', 'error', 'warn', 'info', 'debug', 'debug2')");
    }

    set_log_level(log_level);
}

} // namespace kotekan
