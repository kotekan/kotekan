#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include "restServer.hpp" // for connectionInstance
#include "visUtil.hpp"    // for StatTracker

#include <cstdint> // for uint32_t
#include <map>     // for map
#include <string>  // for string
#include <thread>  // for thread

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
     * Get CPU stat from /proc/stat and stage stat from /proc/self/tid/stat.
     * Thread list maintained and passed by Stage class.
     **/
    void track_cpu();

    /**
     * @brief Get the thead list of all stage tids.
     * 
     * @return a pointer to thread_list.
     **/
    std::map<std::string, std::vector<pid_t>>* get_tid_list();

private:
    std::thread this_thread;
    bool stop_thread;
    std::map<std::string, std::vector<pid_t>> thread_list;
    std::map<std::string, std::vector<CpuStat>> ult_list;
    uint32_t prev_cpu_time;
};

} // namespace kotekan

#endif /* CHIME_HPP */