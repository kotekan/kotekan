#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include "restServer.hpp"

#include "visUtil.hpp"

#include <map>
#include <string>
#include <cstdint>

namespace kotekan {

// Store thread CPU stats
struct CpuStat {
    // StatTracker object is used to get average usage.
    StatTracker utime_usage = StatTracker(120);
    StatTracker stime_usage = StatTracker(120);

    uint32_t prev_utime = 0;
    uint32_t prev_stime = 0;
};

/**
 * @class CpuMonitor
 *
 * @brief Create a new thread to keep tracking all stage cpu usage.
 * This implementation is only applicable on Linux.
 **/
class CpuMonitor {
public:
    CpuMonitor();
    ~CpuMonitor();

    /**
     * @brief Create a new thread and start tracking.
     **/
    void start();
    void stop();

    /**
     * @brief Give stage CPU usage in json format.
     **/
    void cpu_ult_call_back(connectionInstance& conn);

    /**
     * @brief Compute stage CPU usage periodically (thread entry function).
     * Get CPU stat from /proc/stat and stage stat from /proc/self/<tid>/stat.
     * Thread list maintained and passed by Stage class.
     **/
    static void* track_cpu(void *);

private:
    static bool stop_thread;
    static std::map<std::string, CpuStat> ult_list;
    static uint32_t prev_cpu_time;
};

} // namespace kotekan

#endif /* CHIME_HPP */
