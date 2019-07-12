#include "errors.h"

// Default values for log levels.
int __log_level = 3;
int __enable_syslog = 1;
const int __max_log_msg_len = 1024;

enum ReturnCode __status_code = CLEAN_EXIT;
const char *returnCodeNames[RETURN_CODE_COUNT] = {"CLEAN_EXIT", "FATAL_ERROR", "TEST_PASSED", "TEST_FAILED"};

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

// Starts kotekan shutdown and sets a status code. 
void exit_kotekan(enum ReturnCode code)
{
  __status_code = code;
  raise(SIGINT); 
}

enum ReturnCode get_exit_code() {return __status_code;}

// Return error code as string
char * get_exit_code_string(enum ReturnCode code) {

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
      INFO("status_code: %d", code);
      return "INVALID_CODE";
  }

}
