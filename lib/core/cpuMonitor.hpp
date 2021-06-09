#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include "restServer.hpp"

#include <map>
#include <string>
#include <cstdint>

namespace kotekan {

struct CpuStat {
    double utime_usage = 0;
    double stime_usage = 0;
    uint32_t prev_utime = 0;
    uint32_t prev_stime = 0;
};

class CpuMonitor {
public:
    CpuMonitor();

    void start();

    void cpu_ult_call_back(connectionInstance& conn);

    static void* track_cpu(void *);

private:
    // List of CPU usage data <stage_name, CPU_stat>
    static std::map<std::string, CpuStat> ult_list;

    static uint32_t prev_cpu_time;
};

} // namespace kotekan

#endif /* CHIME_HPP */
