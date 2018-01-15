#include <pthread.h>
#include <sched.h>
#include "errors.h"
#include <syslog.h>
#include <stdarg.h>

#include "KotekanProcess.hpp"

KotekanProcess::KotekanProcess(Config &config, const string& unique_name,
                bufferContainer &buffer_container_,
                std::function<void(const KotekanProcess&)> main_thread_ref) :
    stop_thread(false), config(config),
    unique_name(unique_name),
    buffer_container(buffer_container_),
    this_thread(), main_thread_fn(main_thread_ref) {

    set_cpu_affinity(config.get_int_array(unique_name, "cpu_affinity"));

    // Set the local log level.
    string s_log_level = config.get_string(unique_name, "log_level");
    logLevel log_level;

    if (strcasecmp(s_log_level.c_str(), "off") == 0) {
        log_level = logLevel::OFF;
    } else if (strcasecmp(s_log_level.c_str(), "error") == 0) {
        log_level = logLevel::ERROR;
    } else if (strcasecmp(s_log_level.c_str(), "warn") == 0) {
        log_level = logLevel::WARN;
    } else if (strcasecmp(s_log_level.c_str(), "info") == 0) {
        log_level = logLevel::INFO;
    } else if (strcasecmp(s_log_level.c_str(), "debug") == 0) {
        log_level = logLevel::DEBUG;
    } else if (strcasecmp(s_log_level.c_str(), "debug2") == 0) {
        log_level = logLevel::DEBUG2;
    } else {
        throw std::runtime_error("The value given for log_level: '" + s_log_level + "is not valid! " +
                "(It should be one of 'off', 'error', 'warn', 'info', 'debug', 'debug2')");
    }

    __log_level = static_cast<std::underlying_type<logLevel>::type>(log_level);
    INFO("Set log level to %d", __log_level);
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

#ifndef MAC_OSX
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    INFO("Setting thread affinity");
    for (auto &i : cpu_affinity)
        CPU_SET(i, &cpuset);

    err = pthread_setaffinity_np(this_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
#endif

    // Need to add thread name
    if (err)
        ERROR("Failed to set thread affinity for ..., error code %d", err);
}

void KotekanProcess::set_cpu_affinity(const std::vector<int> &cpu_affinity_) {
    {
        std::lock_guard<std::mutex> lock(cpu_affinity_lock);
        cpu_affinity = cpu_affinity_;
    }
    apply_cpu_affinity();
}

void KotekanProcess::internal_logging(int type, const char* format, ...) {
    const int max_log_msg_len = 1024;
    char log_buf[max_log_msg_len];
    va_list args;
    va_start(args, format);
    vsnprintf(log_buf, max_log_msg_len, format, args);
    va_end(args);
    syslog(type, "%s: %s", unique_name.c_str(), log_buf);
}

void Error(const char* format, ...)
{

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

