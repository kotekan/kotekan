#ifndef LOG_BUFFER_H
#define LOG_BUFFER_H

// The most recent log lines, kept in memory so the /logs endpoint can serve them.
// Off (and empty) until set_log_buffer() turns it on.

/// The number of lines kept. Each is cut to MAX_LOG_MSG_LEN, so this bounds the memory used.
#define LOG_BUFFER_LINES 1000

#ifdef __cplusplus
extern "C" {
#endif

/// Records one log line. Does nothing while the buffer is off.
void log_buffer_append(int level, const char* prefix, const char* msg);

#ifdef __cplusplus
}

#include "json.hpp" // for json

namespace kotekan {

/// Turns the buffer on or off. Turning it off drops the kept lines.
void set_log_buffer(bool on);

/// Whether the buffer is on.
bool log_buffer_on();

/// The kept lines, oldest first, as a list of {"time", "level", "msg"} objects.
nlohmann::json get_log_buffer();

} // namespace kotekan
#endif

#endif
