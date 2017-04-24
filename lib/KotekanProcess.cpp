#include <pthread.h>
#include <sched.h>
#include "errors.h"

#include "KotekanProcess.hpp"

KotekanProcess::KotekanProcess(Config &config,
                std::function<void(const KotekanProcess&)> main_thread_ref) :
    stop_thread(false), config(config),
    this_thread(), main_thread_fn(main_thread_ref) {
}

void KotekanProcess::start() {
    this_thread = std::thread(main_thread_fn, std::ref(*this));

    // Requires Linux, this could possibly be made more general someday.
    // TODO Move to config
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    INFO("Setting thread affinity");
    for (int j = 4; j < 12; j++)
        CPU_SET(j, &cpuset);
    pthread_setaffinity_np(this_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
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

