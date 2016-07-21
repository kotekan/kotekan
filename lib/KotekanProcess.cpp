#include "KotekanProcess.h"

KotekanProcess::KotekanProcess(struct Config &config,
                std::function<void(const KotekanProcess&)> main_thread_ref) :
    stop_thread(false), config(config),
    this_thread(), main_thread_fn(main_thread_ref) {

}

void KotekanProcess::start() {
    this_thread = std::thread(main_thread_fn, std::ref(*this));
}

void KotekanProcess::main_thread() {}

KotekanProcess::~KotekanProcess() {
    stop_thread = true;
    if (this_thread.joinable())
        this_thread.join();
}

