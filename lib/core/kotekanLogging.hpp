#ifndef KOTEKAN_LOGGING_H
#define KOTEKAN_LOGGING_H

#include "errors.h"

#include "fmt.hpp"

#include <stdarg.h>
#include <string>
#include <syslog.h>

using std::string;

namespace kotekan {

// Note that the macros support fmt's python style string formatting only.

// The deprecated macros with a `_F` suffix are to be used in C code only and only offer
// printf-style string formatting. They can be found in errors.h.

// Log_level
// 0 = OFF (No logs at all)
// 1 = ERROR (Serious error)
// 2 = WARN (Warning about something wrong)
// 3 = INFO (Helpful ideally short and infrequent, message about system status)
// 4 = DEBUG (Message for debugging reasons only)
// 5 = DEBUG2 (Super detailed debugging messages)
// Note both DEBUG and DEBUG2 are removed entirely when building in release mode.
enum class logLevel { OFF = 0, ERROR = 1, WARN = 2, INFO = 3, DEBUG = 4, DEBUG2 = 5 };

// Macro to pass a string and arguments to fmt::format including a compile-time string format check.
#define FORMAT(m, a...) fmt::format(FMT_STRING(m), ##a)

// These macros check if the given value evaluates to True and if so report an error and exit
// kotekan.
#define CHECK_ERROR(err)                                                                           \
    if (err) {                                                                                     \
        kotekanLogging::internal_logging(LOG_ERR, __log_prefix,                                    \
                                         fmt("Error at {:s}:{:d}; Error type: {:s}"), __FILE__,    \
                                         __LINE__, strerror(errno));                               \
        exit(errno);                                                                               \
    }
#define CHECK_MEM(pointer)                                                                         \
    if (pointer == NULL) {                                                                         \
        internal_logging(LOG_ERR, __log_prefix, fmt("Error at {:s}:{:d}; Null pointer! "),         \
                         __FILE__, __LINE__);                                                      \
        exit(-1);                                                                                  \
    }

#ifdef DEBUGGING
#define DEBUG(m, a...)                                                                             \
    if (_member_log_level > 3) {                                                                   \
        internal_logging(LOG_DEBUG, __log_prefix, fmt(m), ##a);                                    \
    }
#define DEBUG_NON_OO(m, a...)                                                                      \
    if (_global_log_level > 3) {                                                                   \
        kotekan::kotekanLogging::internal_logging(LOG_DEBUG, "", fmt(m), ##a);                     \
    }

#define DEBUG2(m, a...)                                                                            \
    if (_member_log_level > 4) {                                                                   \
        internal_logging(LOG_DEBUG, __log_prefix, fmt(m), ##a);                                    \
    }
#define DEBUG2_NON_OO(m, a...)                                                                     \
    if (_global_log_level > 4) {                                                                   \
        kotekan::kotekanLogging::internal_logging(LOG_DEBUG, "", fmt(m), ##a);                     \
    }
#else

// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG(m, a...) (void)0;         // No op.
#define DEBUG_NON_OO(m, a...) (void)0;  // No op.

// Use this for extra verbose messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG2(m, a...) (void)0;        // No op.
#define DEBUG2_NON_OO(m, a...) (void)0; // No op.

#endif

// Use this for serious errors.  i.e. things that require the program to end.
// Always prints, no check for log level
#define ERROR(m, a...)                                                                             \
    if (_member_log_level > 0) {                                                                   \
        internal_logging(LOG_ERR, __log_prefix, fmt(m), ##a);                                      \
    }
#define ERROR_NON_OO(m, a...)                                                                      \
    if (_global_log_level > 0) {                                                                   \
        kotekan::kotekanLogging::internal_logging(LOG_ERR, "", fmt(m), ##a);                       \
    }

// This is for errors that could cause problems with the operation, or data issues,
// but don't cause the program to fail.
#define WARN(m, a...)                                                                              \
    if (_member_log_level > 1) {                                                                   \
        internal_logging(LOG_WARNING, __log_prefix, fmt(m), ##a);                                  \
    }
#define WARN_NON_OO(m, a...)                                                                       \
    if (_global_log_level > 1) {                                                                   \
        kotekan::kotekanLogging::internal_logging(LOG_WARNING, "", fmt(m), ##a);                   \
    }

// Useful messages to say what the application is doing.
// Should be used sparingly, and limited to useful areas.
#define INFO(m, a...)                                                                              \
    if (_member_log_level > 2) {                                                                   \
        internal_logging(LOG_INFO, __log_prefix, fmt(m), ##a);                                     \
    }
#define INFO_NON_OO(m, a...)                                                                       \
    if (_global_log_level > 2) {                                                                   \
        kotekan::kotekanLogging::internal_logging(LOG_INFO, "", fmt(m), ##a);                      \
    }

// Use this for fatal errors that kotekan can't recover from.
// Prints an error message and raises a SIGINT.
#define FATAL_ERROR(m, a...)                                                                       \
    {                                                                                              \
        ERROR(m, ##a);                                                                             \
        set_error_message(fmt(m), ##a);                                                            \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
    }
#define FATAL_ERROR_NON_OO(m, a...)                                                                \
    {                                                                                              \
        ERROR_NON_OO(m, ##a);                                                                      \
        kotekan::kotekanLogging::set_error_message(fmt(m), ##a);                                   \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
    }


class kotekanLogging {
public:
    kotekanLogging();

    void set_log_level(const logLevel& log_level);
    void set_log_level(const string& string_log_level);
    void set_log_prefix(const string& log_prefix);

    template<typename... Args>
    static void internal_logging(int type, fmt::basic_string_view<char> log_prefix,
                                 const fmt::basic_string_view<char> format, const Args&... args);

    template<typename... Args>
    static void set_error_message(const fmt::basic_string_view<char> format, const Args&... args);

protected:
    int _member_log_level;
    string __log_prefix;

private:
    static void vinternal_logging(int type, fmt::basic_string_view<char> log_prefix,
                                  const fmt::basic_string_view<char> format, fmt::format_args args);
    static void vset_error_message(const fmt::basic_string_view<char> format,
                                   fmt::format_args args);
};

template<typename... Args>
void kotekanLogging::internal_logging(int type, fmt::basic_string_view<char> log_prefix,
                                      const fmt::basic_string_view<char> format,
                                      const Args&... args) {
    vinternal_logging(type, log_prefix, format, fmt::make_format_args(args...));
}

// Stores the error message
template<typename... Args>
void kotekanLogging::set_error_message(const fmt::basic_string_view<char> format,
                                       const Args&... args) {
    vset_error_message(format, fmt::make_format_args(args...));
}

} // namespace kotekan

#endif /* KOTEKAN_LOGGING_H */
