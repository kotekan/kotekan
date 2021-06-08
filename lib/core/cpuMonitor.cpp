#include "cpuMonitor.hpp"

#include "kotekanLogging.hpp"
#include <pthread.h>
#include <fstream>

using namespace std::chrono_literals;
using namespace std::placeholders;

namespace kotekan {

std::map<std::string, CpuStat> CpuMonitor::ult_list;
std::map<std::string, pid_t> CpuMonitor::thread_list;
uint32_t CpuMonitor::prev_cpu_time = 0;

CpuMonitor::CpuMonitor() {};

void CpuMonitor::start() {
    pthread_t pid;
    pthread_create(&pid, nullptr, CpuMonitor::track_cpu, nullptr);
    pthread_detach(pid);

    // Register CPU usage callback
    restServer::instance().register_get_callback(
        "/cpu_ult", std::bind(&CpuMonitor::cpu_ult_call_back, this, _1));
}

void* CpuMonitor::track_cpu(void *) {
    while (1) {
        // Read total CPU stat from /proc/stat first line
        std::string stat;
        std::ifstream cpu_stat("/proc/stat", std::ifstream::in);
        getline(cpu_stat, stat);
        std::istringstream iss(stat);

        // Parse and get total cpu time
        char prefix[10];
        iss >> prefix;
        uint32_t num = 0;
        uint32_t cpu_time = 0;
        for (int i = 0; i < 10; i++) {
            iss >> num;
            cpu_time += num;
        }

        // Read CPU stat for each thread
        for(auto element : thread_list) {
            char fname[100];
            snprintf(fname, sizeof(fname), "/proc/self/task/%d/stat", element.second);
            FILE *fp = fopen(fname, "r");

            ERROR_NON_OO("Read stage: {:s}, tid: {:d}", element.first, element.second);

            if (!fp) ERROR_NON_OO("Cannot open tid/stat!");

            if (fp) {
                // Get the 14th (utime) and the 15th (stime) numbers
                uint32_t utime = 0, stime = 0;
                fscanf(fp, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %lu %lu", &utime, &stime);
                ERROR_NON_OO("u={:d}, s={:d}", utime, stime);
                auto itr = ult_list.find(element.first);
                if (itr != ult_list.end()) {
                    ERROR_NON_OO("u={:d}, s={:d}, pu={:d}, ps={:d}, ct={:d}, pct={:d}"
                                , utime, stime, itr->second.prev_utime, itr->second.prev_stime,
                                cpu_time, prev_cpu_time);
                    // Compute usr and sys CPU usage
                    itr->second.utime_usage =
                        100 * (utime - itr->second.prev_utime) / (cpu_time - prev_cpu_time);
                    itr->second.stime_usage =
                        100 * (stime - itr->second.prev_stime) / (cpu_time - prev_cpu_time);
                    // Update thread usr and sys time
                    itr->second.prev_utime = utime;
                    itr->second.prev_stime = stime;
                    ERROR_NON_OO("utime= {:03.2f}, stime={:03.2f}", itr->second.utime_usage, itr->second.stime_usage);
                } else {
                    ult_list[element.first].prev_utime = utime;
                    ult_list[element.first].prev_stime = stime;
                }
            }
        }
        ERROR_NON_OO("cpu time: {:d}", cpu_time);
        prev_cpu_time = cpu_time;

        // Check each stage periodically
        std::this_thread::sleep_for(1000ms);
    }
}

void CpuMonitor::record_tid(pid_t tid, std::string thread_name) {
    // Add stage to the thread list for CPU usage tracking
    thread_list[thread_name] = tid;
}

void CpuMonitor::cpu_ult_call_back(connectionInstance& conn) {
    nlohmann::json cpu_ult_json = {};

    for (auto element : ult_list) {
        nlohmann::json thread_cpu_ult = {};
        thread_cpu_ult["usr_cpu_ult"] = element.second.utime_usage;
        thread_cpu_ult["sys_cpu_ult"] = element.second.stime_usage;

        cpu_ult_json[element.first] = thread_cpu_ult;
    }

    conn.send_json_reply(cpu_ult_json);
}

} // namespace kotekan
