#include "gpuEventContainer.hpp"


gpuEventContainer::gpuEventContainer() {
    stopping = false;
    signal_set = false;
}

gpuEventContainer::gpuEventContainer(const gpuEventContainer& obj) {
    (void)obj;
}

gpuEventContainer::~gpuEventContainer() {}

void gpuEventContainer::reset() {
    {
        std::lock_guard<std::mutex> lock(mux);
        unset();
        signal_set = false;
    }
    cond_var.notify_all();
}

void* gpuEventContainer::get_signal(void) {
    std::lock_guard<std::mutex> lock(mux);
    return get();
}

void gpuEventContainer::set_signal(void* sig) {
    {
        std::lock_guard<std::mutex> lock(mux);
        set(sig);
        signal_set = true;
    }
    cond_var.notify_all();
}

int gpuEventContainer::wait_for_signal() {
    // Wait for a signal to be ready
    std::unique_lock<std::mutex> lock(mux);
    while (!signal_set && !stopping) {
        cond_var.wait(lock);
    }

    // This signal hasn't been set, so we must be stopping without anything
    // to wait for.   If stopping is set, and there is a signal_set as well,
    // then we should still wait for the signal before exiting and
    // cleaning up the CL memory space.
    if (!signal_set) {
        return -1;
    }

    // Call the specialized wait function.
    wait();

    return 1;
}

void gpuEventContainer::wait_for_free_slot() {
    // Wait for signal_set == false, which implies that signal isn't set
    std::unique_lock<std::mutex> lock(mux);
    while (signal_set) {
        cond_var.wait(lock);
    }
}

void gpuEventContainer::stop() {
    {
        std::lock_guard<std::mutex> lock(mux);
        stopping = true;
    }
    cond_var.notify_all();
}
