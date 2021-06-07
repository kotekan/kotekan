#include "Stage.hpp"

#include "Config.hpp"          // for Config
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "util.h"              // for string_tail
#include "cpuMonitor.hpp"

#include "fmt.hpp" // for format

#include <algorithm>    // for copy, max
#include <chrono>       // for seconds
#include <cstdlib>      // for abort
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <future>       // for async, future, future_status, future_status::timeout, launch
#include <pthread.h>    // for pthread_setaffinity_np, pthread_setname_np
#include <regex>        // for match_results<>::_Base_type
#include <sched.h>      // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>    // for runtime_error
#include <system_error> // for system_error
#include <thread>       // for thread

namespace kotekan {

Stage::Stage(Config& config, const std::string& unique_name, bufferContainer& buffer_container_,
             std::function<void(const Stage&)> main_thread_ref) :
    stop_thread(false),
    config(config),
    unique_name(unique_name),
    this_thread(),
    buffer_container(buffer_container_),
    main_thread_fn(main_thread_ref) {

    set_cpu_affinity(config.get<std::vector<int>>(unique_name, "cpu_affinity"));

    // Set the local log level.
    std::string s_log_level = config.get<std::string>(unique_name, "log_level");
    set_log_level(s_log_level);
    set_log_prefix(unique_name);

    // Set the timeout for this stage thread to exit
    join_timeout = config.get_default<uint32_t>(unique_name, "join_timeout", 60);
}

struct Buffer* Stage::get_buffer(const std::string& name) {
    // NOTE: Maybe require that the buffer be given in the stage, not
    // just somewhere in the path to the stage.
    std::string buf_name = config.get<std::string>(unique_name, name);
    return buffer_container.get_buffer(buf_name);
}

std::vector<struct Buffer*> Stage::get_buffer_array(const std::string& name) {
    std::vector<struct Buffer*> bufs;

    std::vector<std::string> buf_names = config.get<std::vector<std::string>>(unique_name, name);
    for (auto& buf_name : buf_names) {
        bufs.push_back(buffer_container.get_buffer(buf_name));
    }

    return bufs;
}

void Stage::apply_cpu_affinity() {

    std::lock_guard<std::mutex> lock(cpu_affinity_lock);

    // Don't set the thread affinity if the thread hasn't been created yet.
    if (!this_thread.joinable())
        return;

// TODO Enable this for MACOS Systems as well.
#ifndef MAC_OSX
    int err = 0;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    INFO("Setting thread affinity");
    for (auto& i : cpu_affinity)
        CPU_SET(i, &cpuset);

    // Set affinity
    err = pthread_setaffinity_np(this_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (err)
        ERROR("Failed to set thread affinity for {:s}, error code {:d}", unique_name, err);

    // Set debug name as last 15 chars of the config unique_name
    std::string short_name = string_tail(unique_name, 15);
    pthread_setname_np(this_thread.native_handle(), short_name.c_str());
    if (err)
        ERROR("Failed to set thread name for {:s}, error code {:d}", unique_name, err);
#endif
}

void Stage::set_cpu_affinity(const std::vector<int>& cpu_affinity_) {
    {
        std::lock_guard<std::mutex> lock(cpu_affinity_lock);
        cpu_affinity = cpu_affinity_;
    }
    apply_cpu_affinity();
}

void Stage::start() {
    this_thread = std::thread(main_thread_fn, std::ref(*this));

    // Add stage to the thread list for CPU usage tracking
    char fname[100];
    char stage_name[50];
    strcpy(stage_name, unique_name.c_str());
    snprintf(fname, sizeof(fname), "stage: %s, tid: %d", stage_name, this_thread.native_handle());
    ERROR("Stage.cpp: {:s}", fname);
    CpuMonitor::record_tid(this_thread.native_handle(), unique_name);

    apply_cpu_affinity();
}

std::string Stage::get_unique_name() const {
    return unique_name;
}

void Stage::join() {
    if (this_thread.joinable()) {
        // This has the effect of creating a new thread for each thread join,
        // this isn't exactly optimal, but give we are shutting down anyway it should be fine.
        auto thread_joiner = std::async(std::launch::async, &std::thread::join, &this_thread);
        if (thread_joiner.wait_for(std::chrono::seconds(join_timeout))
            == std::future_status::timeout) {
            ERROR("*** EXIT_FAILURE *** The stage {:s} failed to exit (join thread timeout) after "
                  "{:d} seconds. If the stage needs more time to exit, please set the config value "
                  "`join_timeout` for that kotekan_stage.",
                  unique_name, join_timeout);
            std::abort();
        }
    }
}

void Stage::stop() {
    stop_thread = true;
}

void Stage::main_thread() {}

Stage::~Stage() {
    stop_thread = true;
    if (this_thread.joinable())
        this_thread.join();
}

std::string Stage::dot_string(const std::string& prefix) const {
    return fmt::format("{:s}\"{:s}\" [shape=box, color=darkgreen];\n", prefix, get_unique_name());
}

} // namespace kotekan
