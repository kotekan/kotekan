#ifndef ERRORS
#define ERRORS

#include <errno.h>  // for errno
#include <stdio.h>  // for NULL
#include <syslog.h> // for LOG_ERR, LOG_INFO, LOG_WARNING

#ifdef __cplusplus
#include <cerrno>
#endif

enum ReturnCode {
    CLEAN_EXIT = 0,
    FATAL_ERROR = 1,
    TEST_PASSED,
    TEST_FAILED,
    RETURN_CODE_COUNT,
    DATASET_MANAGER_FAILURE
};

#ifdef __cplusplus
extern "C" {
#endif

// The macros with an `_F` suffix are deprecated. They are to be used in C code only and only
// offer printf-style string formatting. For the C++ logging macros, see kotekanLogging.hpp.

// Log_level
// 0 = OFF (No logs at all)
// 1 = ERROR (Serious error)
// 2 = WARN (Warning about something wrong)
// 3 = INFO (Helpful ideally short and infrequent, message about system status)
// 4 = DEBUG (Message for debugging reasons only)
// 5 = DEBUG2 (Super detailed debugging messages)
// Note both DEBUG and DEBUG2 are removed entirely when building in release mode.
extern int _global_log_level;

// If set to 1 use printf instead of syslog
extern int __enable_syslog;

// to store error messages before exiting with error code
extern char __err_msg[1024];
extern const int __max_log_msg_len;

void internal_logging_f(int log, const char* format, ...);
void exit_kotekan(enum ReturnCode code);
enum ReturnCode get_exit_code();
char* get_exit_code_string(enum ReturnCode code);
char* get_error_message();
void set_error_message_f(const char* format, ...);


// These macros check if the given value evaluates to True and if so report an error and exit
// kotekan.
#define CHECK_ERROR_F(err)                                                                         \
    if (err) {                                                                                     \
        internal_logging_f(LOG_ERR, "Error at %s:%d; Error type: %s", __FILE__, __LINE__,          \
                           strerror(errno));                                                       \
        exit(errno);                                                                               \
    }


#define CHECK_MEM_F(pointer)                                                                       \
    if (pointer == NULL) {                                                                         \
        internal_logging_f(LOG_ERR, "Error at %s:%d; Null pointer! ", __FILE__, __LINE__);         \
        exit(-1);                                                                                  \
    }

#ifdef DEBUGGING

// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG_F(m, a...)                                                                           \
    if (_global_log_level > 3) {                                                                   \
        internal_logging_f(LOG_DEBUG, m, ##a);                                                     \
    }


// Use this for extra verbose messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG2_F(m, a...)                                                                          \
    if (_global_log_level > 4) {                                                                   \
        internal_logging_f(LOG_DEBUG, m, ##a);                                                     \
    }


#else

// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG_F(m, a...) (void)0;  // No op.

// Use this for extra verbose messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG2_F(m, a...) (void)0; // No op.

#endif

// Use this for serious errors.  i.e. things that require the program to end.
// Always prints, no check for log level
#define ERROR_F(m, a...)                                                                           \
    if (_global_log_level > 0) {                                                                   \
        internal_logging_f(LOG_ERR, m, ##a);                                                       \
    }

// This is for errors that could cause problems with the operation, or data issues,
// but don't cause the program to fail.
#define WARN_F(m, a...)                                                                            \
    if (_global_log_level > 1) {                                                                   \
        internal_logging_f(LOG_WARNING, m, ##a);                                                   \
    }

// Useful messages to say what the application is doing.
// Should be used sparingly, and limited to useful areas.
#define INFO_F(m, a...)                                                                            \
    if (_global_log_level > 2) {                                                                   \
        internal_logging_f(LOG_INFO, m, ##a);                                                      \
    }

// Use this for fatal errors that kotekan can't recover from.
// Prints an error message and raises a SIGINT.
// Since ReturnCode is defined as a C++ enum, we have to hard code the exit code to 1 here.
#define FATAL_ERROR_F(m, a...)                                                                     \
    {                                                                                              \
        ERROR_F(m, ##a);                                                                           \
        set_error_message_f(m, ##a);                                                               \
        exit_kotekan(1);                                                                           \
    }

// Exit kotekan after a successful test.
#define TEST_PASSED() exit_kotekan(ReturnCode::TEST_PASSED);

// Exit kotekan after a failed test.
#define TEST_FAILED() exit_kotekan(ReturnCode::TEST_FAILED);

#ifdef __cplusplus
}
#endif

#endif
