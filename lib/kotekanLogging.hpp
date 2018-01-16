#ifndef KOTEKAN_LOGGING_H
#define KOTEKAN_LOGGING_H

#include <string>
#include <syslog.h>
#include <stdarg.h>
using std::string;

enum class logLevel {
    OFF = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4,
    DEBUG2 = 5
};

class kotekanLogging {
public:
    kotekanLogging();

    void set_log_level(const logLevel &log_level);
    void set_log_level(const string &string_log_level);
    void set_log_prefix(const string &log_prefix);
protected:
    void internal_logging(int type, const char * format, ...);

    int __log_level;
    string __log_prefix;
};

#endif /* KOTEKAN_LOGGING_H */