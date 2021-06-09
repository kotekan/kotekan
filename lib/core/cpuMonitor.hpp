#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include "restServer.hpp"

#include "visUtil.hpp"

#include <map>
#include <string>
#include <cstdint>

namespace kotekan {

struct CpuStat {
    StatTracker utime_usage = StatTracker(120);
    StatTracker stime_usage = StatTracker(120);
    uint32_t prev_utime = 0;
    uint32_t prev_stime = 0;
};

class CpuMonitor {
public:
    CpuMonitor();
    ~CpuMonitor();

    void start();
    void stop();

    void cpu_ult_call_back(connectionInstance& conn);

    static void* track_cpu(void *);

private:
    static bool stop_thread;

    // List of CPU usage data <stage_name, CPU_stat>
    static std::map<std::string, CpuStat> ult_list;

    static uint32_t prev_cpu_time;
};

} // namespace kotekan

#endif /* CHIME_HPP */
