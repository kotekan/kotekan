#ifndef KOTEKAN_LOGGING_H
#define KOTEKAN_LOGGING_H

#include "errors.h" // for _global_log_level  // IWYU pragma: keep

#include "fmt.hpp" // for fmt, basic_string_view, make_format_args, FMT_STRING

#include <errno.h>  // for errno
#include <string>   // for string
#include <syslog.h> // for LOG_ERR, LOG_INFO, LOG_WARNING


namespace kotekan {

/**
 * \enum logLevel
 * \brief Log level
 * \note Both DEBUG and DEBUG2 are removed entirely when building in release mode.
 * \note The macros support fmt's python style string formatting only.
 * \note The deprecated macros with a `_F` suffix are to be used in C code only and only offer
 *       printf-style string formatting. They can be found in errors.h.
 */
enum class logLevel {
    OFF = 0,   /*!< No logs at all */
    ERROR = 1, /*!< Serious error */
    WARN = 2,  /*!< Warning about something wrong */
    INFO = 3,  /*!< Helpful ideally short and infrequent, message about system status */
    DEBUG = 4, /*!< Message for debugging reasons only */
    DEBUG2 = 5 /*!< Super detailed debugging messages */
};

// Macro to pass a string and arguments to fmt::format including a compile-time string format check.
#define FORMAT(m, a...) fmt::format(FMT_STRING(m), ##a)

// These macros check if the given value evaluates to True and if so report an error and exit
// kotekan.
#define CHECK_ERROR(err)                                                                           \
    do {                                                                                           \
        if (err) {                                                                                 \
            kotekanLogging::internal_logging(LOG_ERR, __log_prefix,                                \
                                             fmt("Error at {:s}:{:d}; Error type: {:s}"),          \
                                             __FILE__, __LINE__, strerror(errno));                 \
            exit(errno);                                                                           \
        }                                                                                          \
    } while (0)
#define CHECK_MEM(pointer)                                                                         \
    do {                                                                                           \
        if (pointer == nullptr) {                                                                  \
            internal_logging(LOG_ERR, __log_prefix, fmt("Error at {:s}:{:d}; Null pointer"),       \
                             __FILE__, __LINE__);                                                  \
            exit(-1);                                                                              \
        }                                                                                          \
    } while (0)

#ifdef DEBUGGING
#define DEBUG(m, a...)                                                                             \
    do {                                                                                           \
        if (_member_log_level > 3) {                                                               \
            internal_logging(LOG_DEBUG, __log_prefix, fmt(m), ##a);                                \
        }                                                                                          \
    } while (0)
#define DEBUG_NON_OO(m, a...)                                                                      \
    do {                                                                                           \
        if (_global_log_level > 3) {                                                               \
            kotekan::kotekanLogging::internal_logging(LOG_DEBUG, "", fmt(m), ##a);                 \
        }                                                                                          \
    } while (0)

#define DEBUG2(m, a...)                                                                            \
    do {                                                                                           \
        if (_member_log_level > 4) {                                                               \
            internal_logging(LOG_DEBUG, __log_prefix, fmt(m), ##a);                                \
        }                                                                                          \
    } while (0)
#define DEBUG2_NON_OO(m, a...)                                                                     \
    do {                                                                                           \
        if (_global_log_level > 4) {                                                               \
            kotekan::kotekanLogging::internal_logging(LOG_DEBUG, "", fmt(m), ##a);                 \
        }                                                                                          \
    } while (0)
#else

// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG(m, a...)                                                                             \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0) // No op.
#define DEBUG_NON_OO(m, a...)                                                                      \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0) // No op.

// Use this for extra verbose messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG2(m, a...)                                                                            \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0) // No op.
#define DEBUG2_NON_OO(m, a...)                                                                     \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0) // No op.

#endif

// Use this for serious errors.  i.e. things that require the program to end.
// Always prints, no check for log level
#define ERROR(m, a...)                                                                             \
    do {                                                                                           \
        if (_member_log_level > 0) {                                                               \
            internal_logging(LOG_ERR, __log_prefix, fmt(m), ##a);                                  \
        }                                                                                          \
    } while (0)
#define ERROR_NON_OO(m, a...)                                                                      \
    do {                                                                                           \
        if (_global_log_level > 0) {                                                               \
            kotekan::kotekanLogging::internal_logging(LOG_ERR, "", fmt(m), ##a);                   \
        }                                                                                          \
    } while (0)

// This is for errors that could cause problems with the operation, or data issues,
// but don't cause the program to fail.
#define WARN(m, a...)                                                                              \
    do {                                                                                           \
        if (_member_log_level > 1) {                                                               \
            internal_logging(LOG_WARNING, __log_prefix, fmt(m), ##a);                              \
        }                                                                                          \
    } while (0)
#define WARN_NON_OO(m, a...)                                                                       \
    do {                                                                                           \
        if (_global_log_level > 1) {                                                               \
            kotekan::kotekanLogging::internal_logging(LOG_WARNING, "", fmt(m), ##a);               \
        }                                                                                          \
    } while (0)

// Useful messages to say what the application is doing.
// Should be used sparingly, and limited to useful areas.
#define INFO(m, a...)                                                                              \
    do {                                                                                           \
        if (_member_log_level > 2) {                                                               \
            internal_logging(LOG_INFO, __log_prefix, fmt(m), ##a);                                 \
        }                                                                                          \
    } while (0)
#define INFO_NON_OO(m, a...)                                                                       \
    do {                                                                                           \
        if (_global_log_level > 2) {                                                               \
            kotekan::kotekanLogging::internal_logging(LOG_INFO, "", fmt(m), ##a);                  \
        }                                                                                          \
    } while (0)

// Use this for fatal errors that kotekan can't recover from.
// Prints an error message and raises a SIGTERM.
#define FATAL_ERROR(m, a...)                                                                       \
    do {                                                                                           \
        ERROR(m, ##a);                                                                             \
        set_error_message(fmt(m), ##a);                                                            \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
    } while (0)
#define FATAL_ERROR_NON_OO(m, a...)                                                                \
    do {                                                                                           \
        ERROR_NON_OO(m, ##a);                                                                      \
        kotekan::kotekanLogging::set_error_message(fmt(m), ##a);                                   \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
    } while (0)


class kotekanLogging {
public:
    kotekanLogging();

    void set_log_level(const logLevel& log_level);
    void set_log_level(const std::string& string_log_level);
    void set_log_prefix(const std::string& log_prefix);

    template<typename... Args>
    static void internal_logging(int type, fmt::basic_string_view<char> log_prefix,
                                 const fmt::basic_string_view<char> format, const Args&... args);

    template<typename... Args>
    static void set_error_message(const fmt::basic_string_view<char> format, const Args&... args);

protected:
    int _member_log_level;
    std::string __log_prefix;

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
