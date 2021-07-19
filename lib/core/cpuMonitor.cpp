#include "cpuMonitor.hpp"

#include "kotekanLogging.hpp" // for WARN_NON_OO

#include "json.hpp" // for json, basic_json<>::object_t, basic_json<>::value_type

#include <chrono>     // for operator""ms, chrono_literals
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, _Placeholder, bind, _1, placeholders
#include <math.h>     // for floor
#include <pthread.h>  // for pthread_detach, pthread_setaffinity_np
#include <sched.h>    // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdio.h>    // for fclose, fopen, fscanf, snprintf, FILE
#include <unistd.h>   // for pid_t, sysconf, _SC_NPROCESSORS_ONLN
#include <utility>    // for pair
#include <vector>     // for vector

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

        // Get tids from all stages
        std::map<std::string, std::vector<pid_t>> tid_list;
        for (auto stage : stages) {
            tid_list[stage.first] = (stage.second)->get_tids();
        }

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
        for (auto stage : tid_list) {
            for (auto tid : stage.second) {
                char fname[40];
                snprintf(fname, sizeof(fname), "/proc/self/task/%d/stat", tid);
                FILE* thread_fp = fopen(fname, "r");

                if (thread_fp) {
                    // Get the 14th (utime) and the 15th (stime) stats
                    uint32_t utime = 0, stime = 0;
                    ret = fscanf(thread_fp,
                                 "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %u %u",
                                 &utime, &stime);

                    auto stage_itr = ult_list.find(stage.first);
                    if (stage_itr != ult_list.end()) {
                        auto thread_itr = (stage_itr->second).find(tid);
                        if (thread_itr != (stage_itr->second).end()) {
                            if (ret == 2) {
                                // Compute usr and sys CPU usage
                                (thread_itr->second)
                                    .utime_usage.add_sample(
                                        core_num * 100 * (utime - (thread_itr->second).prev_utime)
                                        / (cpu_time - prev_cpu_time));
                                (thread_itr->second)
                                    .stime_usage.add_sample(
                                        core_num * 100 * (stime - (thread_itr->second).prev_stime)
                                        / (cpu_time - prev_cpu_time));
                                // Update thread usr and sys time
                                (thread_itr->second).prev_utime = utime;
                                (thread_itr->second).prev_stime = stime;
                            } else {
                                WARN_NON_OO("CPU monitor read insufficient stats from {:s}", fname);
                            }
                        } else {
                            // Create new thread record
                            (stage_itr->second)[tid].prev_utime = utime;
                            (stage_itr->second)[tid].prev_stime = stime;
                        }
                    } else {
                        // Create new stage and thread record
                        (ult_list[stage.first])[tid].prev_utime = utime;
                        (ult_list[stage.first])[tid].prev_stime = stime;
                    }
                } else {
                    // The thread has been terminated early, add 0 to stats
                    (ult_list[stage.first])[tid].utime_usage.add_sample(0);
                    (ult_list[stage.first])[tid].stime_usage.add_sample(0);
                    WARN_NON_OO("CPU monitor cannot read from {:s}", fname);
                }
                fclose(thread_fp);
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

    for (auto& stage : ult_list) {
        nlohmann::json stage_cpu_ult = {};
        for (auto& thread : stage.second) {
            nlohmann::json thread_cpu_ult = {};
            // Limit outputs to two digits
            thread_cpu_ult["usr_cpu_ult"] =
                floor((thread.second).utime_usage.get_avg() * 100) / 100;
            thread_cpu_ult["sys_cpu_ult"] =
                floor((thread.second).stime_usage.get_avg() * 100) / 100;
            stage_cpu_ult[std::to_string(thread.first)] = thread_cpu_ult;
        }
        cpu_ult_json[stage.first] = stage_cpu_ult;
    }

    conn.send_json_reply(cpu_ult_json);
}

void CpuMonitor::save_stages(std::map<std::string, Stage*> input_stages) {
    stages = input_stages;
}

void CpuMonitor::set_affinity(Config& config) {
    std::vector<int32_t> cpu_affinity =
        config.get<std::vector<int32_t>>("/cpu_monitor", "cpu_affinity");

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto core_id : cpu_affinity)
        CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(this_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
}

} // namespace kotekan
