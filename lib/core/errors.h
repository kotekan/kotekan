#ifndef ERRORS
#define ERRORS

#include <syslog.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <signal.h>

#ifdef __cplusplus
#include <cstring>
#include <cerrno>
using std::strerror;
#endif

enum ReturnCode {CLEAN_EXIT = 0, FATAL_ERROR, TEST_PASSED, TEST_FAILED, RETURN_CODE_COUNT}; 

#ifdef __cplusplus
extern "C" {
#endif

// Log_level
// 0 = OFF (No logs at all)
// 1 = ERROR (Serious error)
// 2 = WARN (Warning about something wrong)
// 3 = INFO (Helpful ideally short and infrequent, message about system status)
// 4 = DEBUG (Message for debugging reasons only)
// 5 = DEBUG2 (Super detailed debugging messages)
// Note both DEBUG and DEBUG2 are removed entirely when building in release mode.
extern int __log_level;

// If set to 1 use printf instead of syslog
extern int __enable_syslog;

extern const int __max_log_msg_len;

void internal_logging(int log, const char * format, ...);
void exit_kotekan(enum ReturnCode code);

#define CHECK_ERROR( err )                                         \
    if ( err ) {                                                    \
        internal_logging(LOG_ERR, "Error at %s:%d; Error type: %s",               \
                __FILE__, __LINE__, strerror(errno));               \
        exit( errno );                                              \
    }

#define CHECK_MEM( pointer )                                  \
    if ( pointer == NULL ) {                                  \
        internal_logging(LOG_ERR, "Error at %s:%d; Null pointer! ",          \
                __FILE__, __LINE__);                          \
        exit( -1 );                                        \
    }

#ifdef DEBUGGING

// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG(m, a...) if (__log_level > 3) { internal_logging(LOG_DEBUG, m, ## a); }

// Use this for extra verbose messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG2(m, a...) if (__log_level > 4) { internal_logging(LOG_DEBUG, m, ## a); }

#else

// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG(m, a...) (void)0; // No op.

// Use this for extra verbose messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
// Requires a build with -DCMAKE_BUILD_TYPE=Debug
#define DEBUG2(m, a...) (void)0; // No op.

#endif

// Use this for serious errors.  i.e. things that require the program to end.
// Always prints, no check for log level
#define ERROR(m, a...) if (__log_level > 0) { internal_logging(LOG_ERR, m, ## a); }

// This is for errors that could cause problems with the operation, or data issues,
// but don't cause the program to fail.
#define WARN(m, a...) if (__log_level > 1) { internal_logging(LOG_WARNING, m, ## a); }

// Useful messages to say what the application is doing.
// Should be used sparingly, and limited to useful areas.
#define INFO(m, a...) if (__log_level > 2) { internal_logging(LOG_INFO, m, ## a); }

// Use this for fatal errors that kotekan can't recover from.
// Prints an error message and raises a SIGINT.
#define FATAL_ERROR(message, a...) { ERROR(message, ## a); exit_kotekan(ReturnCode::FATAL_ERROR);}

#ifdef __cplusplus
}
#endif

#endif
