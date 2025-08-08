#include "errors.h"

#include <signal.h> // for raise, SIGTERM
#include <stdarg.h> // for va_end, va_list, va_start

// Default values for log levels.
int _global_log_level = 3;
int __enable_syslog = 1;
const int __max_log_msg_len = 1024;

enum ReturnCode __status_code = CLEAN_EXIT;
char __err_msg[1024] = "not set";

void internal_logging_f(int log, const char* format, ...) {
    va_list args;
    va_start(args, format);
    if (__enable_syslog == 1) {
        (void)vsyslog(log, format, args);
    } else {
        char log_buf[__max_log_msg_len];
        (void)vsnprintf(log_buf, __max_log_msg_len, format, args);
        fprintf(stderr, "%s: %s\n", get_log_level_string(log), log_buf);
    }
    va_end(args);
}

// Starts kotekan shutdown and sets a status code.
void exit_kotekan(enum ReturnCode code) {
    __status_code = code;
    raise(SIGTERM);
}

enum ReturnCode get_exit_code() {
    return __status_code;
}
char* get_error_message() {
    return __err_msg;
}

// Return log level as string
const char* get_log_level_string(int log) {
    switch (log) {
        case LOG_EMERG:
            return "EMERGENCY";
        case LOG_ALERT:
            return "ALERT";
        case LOG_CRIT:
            return "CRITICAL";
        case LOG_ERR:
            return "ERROR";
        case LOG_WARNING:
            return "WARNING";
        case LOG_NOTICE:
            return "NOTICE";
        case LOG_INFO:
            return "INFO";
        case LOG_DEBUG:
            return "DEBUG";
        default:
            return "(unknown log level)";
    }
}

// Return error code as string
char* get_exit_code_string(enum ReturnCode code) {

    switch (code) {
        case CLEAN_EXIT:
            return "CLEAN_EXIT";
        case FATAL_ERROR:
            return "FATAL_ERROR";
        case TEST_PASSED:
            return "TEST_PASSED";
        case TEST_FAILED:
            return "TEST_FAILED";
        default:
            return "INVALID_CODE";
    }
}

// Stores the error message
void set_error_message_f(const char* format, ...) {
    va_list args;
    va_start(args, format);
    (void)vsnprintf(__err_msg, __max_log_msg_len, format, args);
    va_end(args);
}
