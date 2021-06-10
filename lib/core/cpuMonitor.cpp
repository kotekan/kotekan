#include "cpuMonitor.hpp"

#include "Stage.hpp"
#include "kotekanLogging.hpp"

#include <fstream>
#include <pthread.h>

using namespace std::chrono_literals;
using namespace std::placeholders;

namespace kotekan {

std::map<std::string, CpuStat> CpuMonitor::ult_list;
uint32_t CpuMonitor::prev_cpu_time = 0;
bool CpuMonitor::stop_thread = false;

CpuMonitor::CpuMonitor() {
    // Register CPU usage callback
    restServer::instance().register_get_callback(
        "/cpu_ult", std::bind(&CpuMonitor::cpu_ult_call_back, this, _1));
}

CpuMonitor::~CpuMonitor() {
    restServer::instance().remove_get_callback("/cpu_ult");
    ult_list.clear();
}

void CpuMonitor::start() {
    pthread_t pid;
    pthread_create(&pid, nullptr, CpuMonitor::track_cpu, nullptr);
    pthread_detach(pid);
}

void CpuMonitor::stop() {
    stop_thread = true;
}

void* CpuMonitor::track_cpu(void*) {
    while (!stop_thread) {
        uint32_t cpu_times[10];
        uint32_t cpu_time = 0;

        // Read CPU stats from /proc/stat first line
        std::string stat;
        FILE* cpu_fp = fopen("/proc/stat", "r");
        fscanf(cpu_fp, "%*s %u %u %u %u %u %u %u %u %u %u", &cpu_times[0], &cpu_times[1],
               &cpu_times[2], &cpu_times[3], &cpu_times[4], &cpu_times[5], &cpu_times[6],
               &cpu_times[7], &cpu_times[8], &cpu_times[9]);
        fclose(cpu_fp);

        // Compute total cpu time
        for (int i = 0; i < 10; i++) {
            cpu_time += cpu_times[i];
        }

        // Read each thread stats based on tid
        std::map<std::string, pid_t> thread_list = Stage::get_thread_list();
        for (auto element : thread_list) {
            char fname[40];
            snprintf(fname, sizeof(fname), "/proc/self/task/%d/stat", element.second);
            FILE* thread_fp = fopen(fname, "r");

            if (thread_fp) {
                // Get the 14th (utime) and the 15th (stime) stats
                uint32_t utime = 0, stime = 0;
                fscanf(thread_fp, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %u %u",
                       &utime, &stime);

                auto itr = ult_list.find(element.first);
                if (itr != ult_list.end()) {
                    // Compute usr and sys CPU usage
                    itr->second.utime_usage.add_sample(100 * (utime - itr->second.prev_utime)
                                                       / (cpu_time - prev_cpu_time));
                    itr->second.stime_usage.add_sample(100 * (stime - itr->second.prev_stime)
                                                       / (cpu_time - prev_cpu_time));
                    // Update thread usr and sys time
                    itr->second.prev_utime = utime;
                    itr->second.prev_stime = stime;
                } else {
                    // Add new thead to ult_list
                    ult_list[element.first].prev_utime = utime;
                    ult_list[element.first].prev_stime = stime;
                }
            }
            fclose(thread_fp);
        }
        // Update cpu time
        prev_cpu_time = cpu_time;

        // Wait for next check
        std::this_thread::sleep_for(1000ms);
    }
}

void CpuMonitor::cpu_ult_call_back(connectionInstance& conn) {
    nlohmann::json cpu_ult_json = {};

    for (auto& element : ult_list) {
        nlohmann::json thread_cpu_ult = {};
        // Limit outputs to two digits
        thread_cpu_ult["usr_cpu_ult"] = floor(element.second.utime_usage.get_avg() * 100) / 100;
        thread_cpu_ult["sys_cpu_ult"] = floor(element.second.stime_usage.get_avg() * 100) / 100;

        cpu_ult_json[element.first] = thread_cpu_ult;
    }

    conn.send_json_reply(cpu_ult_json);
}

} // namespace kotekan
