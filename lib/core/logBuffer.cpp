#include "logBuffer.h"

#include "errors.h" // for get_log_level_string, MAX_LOG_MSG_LEN

#include <atomic>  // for atomic
#include <deque>   // for deque
#include <mutex>   // for mutex, lock_guard
#include <string>  // for string
#include <time.h>  // for clock_gettime, timespec, CLOCK_REALTIME
#include <utility> // for move
#include <vector>  // for vector

namespace {

struct log_line {
    double time;
    int level;
    std::string msg;
};

struct log_buffer {
    std::atomic<bool> on{false};
    std::mutex mutex;
    std::deque<log_line> lines;
};

// Never deleted, so that logging during static destruction stays safe.
log_buffer& buffer() {
    static log_buffer* b = new log_buffer;
    return *b;
}

} // namespace

void log_buffer_append(int level, const char* prefix, const char* msg) {
    log_buffer& b = buffer();
    if (!b.on.load(std::memory_order_relaxed))
        return;

    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    log_line line{ts.tv_sec + ts.tv_nsec * 1e-9, level,
                  prefix[0] ? std::string(prefix) + ": " + msg : std::string(msg)};
    if (line.msg.size() > MAX_LOG_MSG_LEN)
        line.msg.resize(MAX_LOG_MSG_LEN);

    std::lock_guard<std::mutex> lock(b.mutex);
    if (!b.on.load(std::memory_order_relaxed))
        return;
    if (b.lines.size() == LOG_BUFFER_LINES)
        b.lines.pop_front();
    b.lines.push_back(std::move(line));
}

namespace kotekan {

void set_log_buffer(bool on) {
    log_buffer& b = buffer();
    std::lock_guard<std::mutex> lock(b.mutex);
    b.on = on;
    b.lines.clear();
}

bool log_buffer_on() {
    return buffer().on;
}

nlohmann::json get_log_buffer() {
    log_buffer& b = buffer();
    std::vector<log_line> lines;
    {
        // Copy out, so logging threads do not wait on the JSON build.
        std::lock_guard<std::mutex> lock(b.mutex);
        lines.assign(b.lines.begin(), b.lines.end());
    }
    nlohmann::json reply = nlohmann::json::array();
    for (const log_line& line : lines)
        reply.push_back(
            {{"time", line.time}, {"level", get_log_level_string(line.level)}, {"msg", line.msg}});
    return reply;
}

} // namespace kotekan
