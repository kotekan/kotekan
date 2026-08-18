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

/// The kind of event reported to a log_event_handler.
enum class log_event { warning, error };

/// A handler notified of WARN and ERROR events, in addition to the message being
/// logged. There is none in production; boost tests install one (see
/// tests/boost/kotekanTestLogging.hpp) so that an error logged by kotekan fails
/// the test.
///
/// This is deliberately a run-time decision. It used to be a compile-time one:
/// the reporting macros below were defined differently depending on whether
/// BOOST_TEST_MODULE was defined when this header was included, which is true in
/// a boost test but false in every library translation unit. That gave every
/// function *defined in a header* that logs two different bodies. Such functions
/// have external linkage and are emitted as weak symbols, so it was an ODR
/// violation, silently resolved by the linker keeping one arbitrary copy: a test
/// could end up with the library's non-throwing copy, so an expected error
/// terminated the test process instead of failing an assertion, or library code
/// could end up with the test's throwing copy. Which one won depended on link
/// order and LTO. Reporting at run time keeps the macros identical everywhere.
using log_event_handler = void (*)(log_event kind, const char* file, int line,
                                   const std::string& message);

/// The installed handler, or null. Prefer report_log_event() to reading this.
inline std::atomic<log_event_handler> log_event_hook{nullptr};

/// Reports a log event to the installed handler, if there is one.
inline void report_log_event(const log_event kind, const char* const file, const int line,
                             const std::string& message) {
    if (const log_event_handler handler = log_event_hook.load(std::memory_order_relaxed))
        handler(kind, file, line, message);
}

} // namespace kotekan

// Report an error/warning to the installed log event handler.
//
// These must expand to the same tokens in every translation unit; see the comment
// on kotekan::log_event_handler above. Checking for a handler first avoids
// formatting the message when there is none, which is the case in production.
#define KTK_REPORT_ERROR(m, ...)                                                                   \
    do {                                                                                           \
        if (kotekan::log_event_hook.load(std::memory_order_relaxed))                               \
            kotekan::report_log_event(kotekan::log_event::error, __FILE__, __LINE__,               \
                                      FORMAT(m, ##__VA_ARGS__));                                   \
    } while (0)
#define KTK_REPORT_WARNING(m, ...)                                                                 \
    do {                                                                                           \
        if (kotekan::log_event_hook.load(std::memory_order_relaxed))                               \
            kotekan::report_log_event(kotekan::log_event::warning, __FILE__, __LINE__,             \
                                      FORMAT(m, ##__VA_ARGS__));                                   \
    } while (0)

// Macro to pass a string and arguments to fmt::format including a compile-time string format check.
#define FORMAT(m, ...) fmt::format(FMT_STRING(m), ##__VA_ARGS__)

// These macros check if the given value evaluates to True and if so report an error and exit
// kotekan.
#define CHECK_ERROR(err)                                                                           \
    do {                                                                                           \
        if (err) {                                                                                 \
            kotekanLogging::internal_logging(LOG_ERR, __log_prefix,                                \
                                             fmt("Error at {:s}:{:d}; Error type: {:s}"),          \
                                             __FILE__, __LINE__, strerror(errno));                 \
            KTK_REPORT_ERROR("Error at {}:{}; Error type: {}", __FILE__, __LINE__,                 \
                             strerror(errno));                                                     \
            exit(errno);                                                                           \
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
        ERROR(m, ##__VA_ARGS__);                                                                   \
        set_error_message(fmt(m), ##__VA_ARGS__);                                                  \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
        throw FatalError(fmt::format(FMT_STRING(m), ##__VA_ARGS__));                               \
    } while (0)
#define FATAL_ERROR_NON_OO(m, ...)                                                                 \
    do {                                                                                           \
        ERROR_NON_OO(m, ##__VA_ARGS__);                                                            \
        kotekan::kotekanLogging::set_error_message(fmt(m), ##__VA_ARGS__);                         \
        exit_kotekan(ReturnCode::FATAL_ERROR);                                                     \
        throw FatalError(fmt::format(FMT_STRING(m), ##__VA_ARGS__));                               \
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
