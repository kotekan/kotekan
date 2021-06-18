#include "cpuMonitor.hpp"

#include "Stage.hpp" // for Stage

#include "json.hpp" // for basic_json<>::object_t, json, basic_json<>::value_type

#include <chrono>      // for operator""ms, chrono_literals
#include <functional>  // for _Bind_helper<>::type, _Placeholder, bind, _1, placeholders
#include <math.h>      // for floor
#include <pthread.h>   // for pthread_detach
#include <stdio.h>     // for fclose, fopen, fscanf, snprintf, FILE
#include <sys/types.h> // for pid_t
#include <utility>     // for pair

using namespace std::chrono_literals;
using namespace std::placeholders;

namespace kotekan {

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
    this_thread = std::thread(&CpuMonitor::track_cpu, this);
    pthread_detach(this_thread.native_handle());
}

void CpuMonitor::stop() {
    stop_thread = true;
}

void CpuMonitor::track_cpu() {
    int core_num = sysconf(_SC_NPROCESSORS_ONLN);
    while (!stop_thread) {
        uint32_t cpu_times[10];
        uint32_t cpu_time = 0;

        // Read CPU stats from /proc/stat first line
        std::string stat;
        FILE* cpu_fp = fopen("/proc/stat", "r");
        int ret = fscanf(cpu_fp, "%*s %u %u %u %u %u %u %u %u %u %u", &cpu_times[0], &cpu_times[1],
                         &cpu_times[2], &cpu_times[3], &cpu_times[4], &cpu_times[5], &cpu_times[6],
                         &cpu_times[7], &cpu_times[8], &cpu_times[9]);
        fclose(cpu_fp);

        // Compute total cpu time
        if (ret == 10) {
            for (int i = 0; i < 10; i++) {
                cpu_time += cpu_times[i];
            }
        } else {
            WARN_NON_OO("CPU monitor read insufficient stats from /proc/stat");
        }

        // Read each thread stats based on tid
        for (auto element : thread_list) {
            int count = 0;
            for (auto tid : element.second) {
                char fname[40];
                snprintf(fname, sizeof(fname), "/proc/self/task/%d/stat", tid);
                FILE* thread_fp = fopen(fname, "r");

                if (thread_fp) {
                    // Get the 14th (utime) and the 15th (stime) stats
                    uint32_t utime = 0, stime = 0;
                    ret = fscanf(thread_fp, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %u %u",
                                &utime, &stime);

                    auto itr = ult_list.find(element.first);
                    if (itr != ult_list.end()) {
                        if (ret == 2) {
                            // Compute usr and sys CPU usage
                            (itr->second)[count].utime_usage.add_sample(core_num * 100 * (utime - (itr->second)[count].prev_utime)
                                                            / (cpu_time - prev_cpu_time));
                            (itr->second)[count].stime_usage.add_sample(core_num * 100 * (stime - (itr->second)[count].prev_stime)
                                                            / (cpu_time - prev_cpu_time));
                            // Update thread usr and sys time
                            (itr->second)[count].prev_utime = utime;
                            (itr->second)[count].prev_stime = stime;
                        } else {
                            WARN_NON_OO("CPU monitor read insufficient stats from {:s}", fname);
                        }
                    } else {
                        // Add new thead to ult_list
                        (ult_list[element.first])[count].prev_utime = utime;
                        (ult_list[element.first])[count].prev_stime = stime;
                    }
                } else {
                    WARN_NON_OO("CPU monitor cannot read from {:s}", fname);
                }
                fclose(thread_fp);
                count++;
            }
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
        nlohmann::json stage_cpu_ult = {};
        int count = 0;
        for (auto& thread : element.second) {
            nlohmann::json thread_cpu_ult = {};
            // Limit outputs to two digits
            thread_cpu_ult["usr_cpu_ult"] = floor(thread.utime_usage.get_avg() * 100) / 100;
            thread_cpu_ult["sys_cpu_ult"] = floor(thread.stime_usage.get_avg() * 100) / 100;
            stage_cpu_ult[count] = thread_cpu_ult;
            count++;
        }
        cpu_ult_json[element.first] = stage_cpu_ult;
    }

    conn.send_json_reply(cpu_ult_json);
}

std::map<std::string, std::vector<pid_t>>* CpuMonitor::get_tid_list() {
    return &thread_list;
}

} // namespace kotekan
