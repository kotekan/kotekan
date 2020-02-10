#include "errors.h"

#include <signal.h> // for raise, SIGINT
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
        fprintf(stderr, "%s\n", log_buf);
    }
    va_end(args);
}

// Starts kotekan shutdown and sets a status code.
void exit_kotekan(enum ReturnCode code) {
    __status_code = code;
    raise(SIGINT);
}

enum ReturnCode get_exit_code() {
    return __status_code;
}
char* get_error_message() {
    return __err_msg;
}

// Return error code as string
char* get_exit_code_string(enum ReturnCode code) {

    switch (code) {
        case CLEAN_EXIT:
            return "CLEAN_EXIT";
            break;
        case FATAL_ERROR:
            return "FATAL_ERROR";
            break;
        case TEST_PASSED:
            return "TEST_PASSED";
            break;
        case TEST_FAILED:
            return "TEST_FAILED";
            break;
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
