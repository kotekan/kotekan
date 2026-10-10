#ifndef KOTEKAN_LOGGING_H
#define KOTEKAN_LOGGING_H

#include "errors.h" // for _global_log_level  // IWYU pragma: keep

#include "fmt.hpp" // for fmt, basic_string_view, FMT_STRING, format_args, make_format_args

#include <atomic>    // for atomic
#include <errno.h>   // for errno
#include <stdexcept> // for runtime_error
#include <string>    // for string, basic_string
#include <syslog.h>  // for LOG_DEBUG, LOG_ERR, LOG_INFO, LOG_WARNING

class FatalError : public std::runtime_error {
public:
    explicit FatalError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

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

/// A handler notified of ERROR and WARN events, in addition to the message being
/// logged. There is none in production; boost tests install one (see
/// tests/boost/kotekanLoggingFixture.hpp) so that an error logged by kotekan fails
/// the test.
using log_event_handler = void (*)(logLevel level, const char* file, int line,
                                   const std::string& message);

/// The installed handler, or null. Prefer report_log_event() to reading this.
inline std::atomic<log_event_handler> log_event_hook{nullptr};

/// Reports a log event to the installed handler, if there is one.
///
/// The message is formatted with fmt::vformat, like
/// kotekanLogging::internal_logging: the format string is passed as a plain
/// string_view, so it is not checked against the argument types at compile time.
template<typename... Args>
inline void report_log_event(const logLevel level, const char* const file, const int line,
                             const fmt::basic_string_view<char> format, const Args&... args) {
    if (const log_event_handler handler = log_event_hook.load(std::memory_order_relaxed))
        handler(level, file, line, fmt::vformat(format, fmt::make_format_args(args...)));
}

} // namespace kotekan

// Report an error/warning to the installed log event handler.
//
// These must expand to the same tokens in every translation unit, or a function
// defined in a header that logs gets two different bodies -- an ODR violation.
// tools/lint.sh enforces that. The handler check is in the macro so that the
// arguments are not evaluated when there is none, as in production.
#define KTK_REPORT_ERROR(m, ...)                                                                   \
    (kotekan::log_event_hook.load(std::memory_order_relaxed)                                       \
         ? kotekan::report_log_event(kotekan::logLevel::ERROR, __FILE__, __LINE__, fmt(m),         \
                                     ##__VA_ARGS__)                                                \
         : (void)0)
#define KTK_REPORT_WARNING(m, ...)                                                                 \
    (kotekan::log_event_hook.load(std::memory_order_relaxed)                                       \
         ? kotekan::report_log_event(kotekan::logLevel::WARN, __FILE__, __LINE__, fmt(m),          \
                                     ##__VA_ARGS__)                                                \
         : (void)0)

// Macro to pass a string and arguments to fmt::format including a compile-time string format check.
#define FORMAT(m, ...) fmt::format(FMT_STRING(m), ##__VA_ARGS__)

// These macros check if the given value evaluates to True and if so report an error and exit
// kotekan.
#define CHECK_ERROR(err)                                                                           \
    do {                                                                                           \
        if (err) {                                                                                 \
            const int ktk_errno = errno;                                                           \
            kotekanLogging::internal_logging(LOG_ERR, __log_prefix,                                \
                                             fmt("Error at {:s}:{:d}; Error type: {:s}"),          \
                                             __FILE__, __LINE__, strerror(ktk_errno));             \
            KTK_REPORT_ERROR("Error at {}:{}; Error type: {}", __FILE__, __LINE__,                 \
                             strerror(ktk_errno));                                                 \
            exit(ktk_errno);                                                                       \
        }                                                                                          \
    } while (0)
#define CHECK_MEM(pointer)                                                                         \
    do {                                                                                           \
        if (pointer == nullptr) {                                                                  \
            internal_logging(LOG_ERR, __log_prefix, fmt("Error at {:s}:{:d}; Null pointer"),       \
                             __FILE__, __LINE__);                                                  \
            KTK_REPORT_ERROR("Error at {}:{}; Null pointer", __FILE__, __LINE__);                  \
            exit(-1);                                                                              \
        }                                                                                          \
    } while (0)

// DEBUG / DEBUG2
// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be compiled out in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#ifdef DEBUGGING
#define DEBUG(m, ...)                                                                              \
    do {                                                                                           \
        if (_member_log_level > 3)                                                                 \
            internal_logging(LOG_DEBUG, __log_prefix, fmt(m), ##__VA_ARGS__);                      \
    } while (0)
#define DEBUG2(m, ...)                                                                             \
    do {                                                                                           \
        if (_member_log_level > 4)                                                                 \
            internal_logging(LOG_DEBUG, __log_prefix, fmt(m), ##__VA_ARGS__);                      \
    } while (0)
#define DEBUG_NON_OO(m, ...)                                                                       \
    do {                                                                                           \
        if (_global_log_level > 3)                                                                 \
            kotekan::kotekanLogging::internal_logging(LOG_DEBUG, "", fmt(m), ##__VA_ARGS__);       \
    } while (0)
#define DEBUG2_NON_OO(m, ...)                                                                      \
    do {                                                                                           \
        if (_global_log_level > 4)                                                                 \
            kotekan::kotekanLogging::internal_logging(LOG_DEBUG, "", fmt(m), ##__VA_ARGS__);       \
    } while (0)
#else // !DEBUGGING
#define DEBUG(m, ...)                                                                              \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0)
#define DEBUG2(m, ...)                                                                             \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0)
#define DEBUG_NON_OO(m, ...)                                                                       \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0)
#define DEBUG2_NON_OO(m, ...)                                                                      \
    do {                                                                                           \
        (void)0;                                                                                   \
    } while (0)
#endif // DEBUGGING

// Use this for fatal errors that need to exit immediately.
// Prints an error message and immediately calls exit().
#define EXIT_ERROR(m, ...)                                                                         \
    do {                                                                                           \
        ERROR(m, ##__VA_ARGS__);                                                                   \
        std::exit(ReturnCode::FATAL_ERROR);                                                        \
    } while (0)
#define EXIT_ERROR_NON_OO(m, ...)                                                                  \
    do {                                                                                           \
        ERROR_NON_OO(m, ##__VA_ARGS__);                                                            \
        std::exit(ReturnCode::FATAL_ERROR);                                                        \
    } while (0)

// Use this for fatal errors that kotekan can't recover from. May shut down gracefully.
// Prints an error message, raises a SIGTERM, and throws (caught for stages)
#define FATAL_ERROR(m, ...)                                                                        \
    do {                                                                                           \
        const std::string _fatal_msg = FORMAT(m, ##__VA_ARGS__);                                   \
        ERROR("{:s}", _fatal_msg);                                                                 \
        set_error_message(fmt("{:s}"), _fatal_msg);                                                \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
        throw FatalError(_fatal_msg);                                                              \
    } while (0)
#define FATAL_ERROR_NON_OO(m, ...)                                                                 \
    do {                                                                                           \
        const std::string _fatal_msg = FORMAT(m, ##__VA_ARGS__);                                   \
        ERROR_NON_OO("{:s}", _fatal_msg);                                                          \
        kotekan::kotekanLogging::set_error_message(fmt("{:s}"), _fatal_msg);                       \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
        throw FatalError(_fatal_msg);                                                              \
    } while (0)


// Use this for serious errors that are guaranteed to cause issues with operation.
// Always prints, no check for log level
#define ERROR(m, ...)                                                                              \
    do {                                                                                           \
        if (_member_log_level > 0)                                                                 \
            internal_logging(LOG_ERR, __log_prefix, fmt(m), ##__VA_ARGS__);                        \
        KTK_REPORT_ERROR(m, ##__VA_ARGS__);                                                        \
    } while (0)
#define ERROR_NON_OO(m, ...)                                                                       \
    do {                                                                                           \
        if (_global_log_level > 0)                                                                 \
            kotekan::kotekanLogging::internal_logging(LOG_ERR, "", fmt(m), ##__VA_ARGS__);         \
        KTK_REPORT_ERROR(m, ##__VA_ARGS__);                                                        \
    } while (0)

// This is for errors that could cause problems with the operation, or data issues,
// but don't cause the program to fail.
#define WARN(m, ...)                                                                               \
    do {                                                                                           \
        if (_member_log_level > 1)                                                                 \
            internal_logging(LOG_WARNING, __log_prefix, fmt(m), ##__VA_ARGS__);                    \
        KTK_REPORT_WARNING(m, ##__VA_ARGS__);                                                      \
    } while (0)
#define WARN_NON_OO(m, ...)                                                                        \
    do {                                                                                           \
        if (_global_log_level > 1)                                                                 \
            kotekan::kotekanLogging::internal_logging(LOG_WARNING, "", fmt(m), ##__VA_ARGS__);     \
        KTK_REPORT_WARNING(m, ##__VA_ARGS__);                                                      \
    } while (0)

// Useful messages to say what the application is doing.
// Should be used sparingly, and limited to useful areas.
#define INFO(m, ...)                                                                               \
    do {                                                                                           \
        if (_member_log_level > 2)                                                                 \
            internal_logging(LOG_INFO, __log_prefix, fmt(m), ##__VA_ARGS__);                       \
    } while (0)
#define INFO_NON_OO(m, ...)                                                                        \
    do {                                                                                           \
        if (_global_log_level > 2)                                                                 \
            kotekan::kotekanLogging::internal_logging(LOG_INFO, "", fmt(m), ##__VA_ARGS__);        \
    } while (0)

namespace kotekan {

class kotekanLogging {
public:
    kotekanLogging();

    void set_log_level(const logLevel& log_level);
    void set_log_level(const std::string& string_log_level);
    void set_log_prefix(const std::string& log_prefix);

    logLevel get_log_level() const;

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
