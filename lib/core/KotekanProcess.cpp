#include <pthread.h>
#include <sched.h>
#include <syslog.h>

#include "KotekanProcess.hpp"
#include "errors.h"
#include "util.h"

KotekanProcess::KotekanProcess(Config &config, const string& unique_name,
                bufferContainer &buffer_container_,
                std::function<void(const KotekanProcess&)> main_thread_ref) :
    stop_thread(false), config(config),
    unique_name(unique_name),
    this_thread(),
    main_thread_fn(main_thread_ref),
    buffer_container(buffer_container_) {

    set_cpu_affinity(config.get_int_array(unique_name, "cpu_affinity"));

    // Set the local log level.
    string s_log_level = config.get_string(unique_name, "log_level");
    set_log_level(s_log_level);
    set_log_prefix(unique_name);
}

struct Buffer* KotekanProcess::get_buffer(const std::string& name) {
    // NOTE: Maybe require that the buffer be given in the process, not
    // just somewhere in the path to the process.
    string buf_name = config.get_string(unique_name, name);
    return buffer_container.get_buffer(buf_name);
}

void KotekanProcess::apply_cpu_affinity() {
    int err = 0;

    std::lock_guard<std::mutex> lock(cpu_affinity_lock);

    // Don't set the thread affinity if the thread hasn't been created yet.
    if(!this_thread.joinable())
        return;

// TODO Enable this for MACOS Systems as well.
#ifndef MAC_OSX
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    INFO("Setting thread affinity");
    for (auto &i : cpu_affinity)
        CPU_SET(i, &cpuset);

    // Set affinity
    err = pthread_setaffinity_np(this_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (err)
        ERROR("Failed to set thread affinity for %s, error code %d", unique_name.c_str(), err);

    // Set debug name as last 15 chars of the config unique_name
    std::string short_name = string_tail(unique_name, 15);
    pthread_setname_np(this_thread.native_handle(), short_name.c_str());
    if (err)
        ERROR("Failed to set thread name for %s, error code %d", unique_name.c_str(), err);
#endif
}

void KotekanProcess::set_cpu_affinity(const std::vector<int> &cpu_affinity_) {
    {
        std::lock_guard<std::mutex> lock(cpu_affinity_lock);
        cpu_affinity = cpu_affinity_;
    }
    apply_cpu_affinity();
}

void KotekanProcess::start() {
    this_thread = std::thread(main_thread_fn, std::ref(*this));

    apply_cpu_affinity();
}

void KotekanProcess::join() {
    if (this_thread.joinable())
        this_thread.join();
}

void KotekanProcess::stop() {
    stop_thread = true;
}

void KotekanProcess::main_thread() {}

KotekanProcess::~KotekanProcess() {
    stop_thread = true;
    if (this_thread.joinable())
        this_thread.join();
}

