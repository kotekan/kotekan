
#ifndef ERRORS
#define ERRORS

// TODO set this via cmake build type!
#define DEBUGING 0

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

#include <syslog.h>

#ifdef __cplusplus
extern "C" {
#endif

char* oclGetOpenCLErrorCodeStr(cl_int input);

extern int log_level_warn;
extern int log_level_debug;
extern int log_level_info;

#define CHECK_CL_ERROR( err )                                      \
    if ( err ) {                                                    \
        syslog(LOG_ERR, "Error at %s:%d; Error type: %s",               \
                __FILE__, __LINE__, oclGetOpenCLErrorCodeStr(err)); \
        exit( err );                                                \
    }

#define CHECK_ERROR( err )                                         \
    if ( err ) {                                                    \
        syslog(LOG_ERR, "Error at %s:%d; Error type: %s",               \
                __FILE__, __LINE__, strerror(errno));               \
        exit( errno );                                              \
    }

#define CHECK_MEM( pointer )                                  \
    if ( pointer == NULL ) {                                  \
        syslog(LOG_ERR, "Error at %s:%d; Null pointer! ",          \
                __FILE__, __LINE__);                          \
        exit( -1 );                                        \
    }

#ifdef DEBUGING

// Use this for messages that shouldn't be shown in the release version.
// This is mostly for testing, tracking down bugs.  It can live in most critical
// sections, since it will be removed in a release build.
#define DEBUG(m, a...) if (log_level_debug) { syslog(LOG_DEBUG, m, ## a); }

#else

#define DEBUG(m, a...) (void)0; // No op.

#endif

// Use this for serious errors.  i.e. things that require the program to end.
#define ERROR(m, a...) syslog(LOG_ERR, m, ## a);

// This is for errors that could cause problems with the operation, or data issues,
// but don't cause the program to fail.
#define WARN(m, a...) if (log_level_warn) { syslog(LOG_WARNING, m, ## a); }

// Useful messages to say what the application is doing.
// Should be used sparingly, and limited to useful areas.
#define INFO(m, a...) if (log_level_info) { syslog(LOG_INFO, m, ## a); }

#ifdef __cplusplus
}
#endif

#endif
