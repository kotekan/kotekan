#include "errors.h"

// Default values for log levels.
int __log_level = 3;
int __enable_syslog = 1;
const int __max_log_msg_len = 1024;

void internal_logging(int log, const char * format, ...)
{
    va_list args;
    va_start(args, format);
    if (__enable_syslog == 1) {
        (void) vsyslog(log, format, args);
    } else {
        char log_buf[__max_log_msg_len];
        (void) vsnprintf(log_buf, __max_log_msg_len, format, args);
        fprintf(stderr, "%s\n", log_buf);
    }
    va_end(args);
}

