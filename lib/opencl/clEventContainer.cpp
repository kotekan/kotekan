#include "clEventContainer.hpp"
#include "errors.h"
#include <unistd.h>


clEventContainer::clEventContainer() {
    reset();
    stopping = false;
}

clEventContainer::clEventContainer(const clEventContainer& obj) {
}

clEventContainer::~clEventContainer() {
}

void clEventContainer::reset() {
    {
        std::lock_guard<std::mutex> lock(mux);
        signal_set = false;
        signal = NULL;
    }
    cond_var.notify_all();
}

void clEventContainer::set_signal(cl_event sig) {
    {
        std::lock_guard<std::mutex> lock(mux);
        signal = sig;
        signal_set = true;
    }
    cond_var.notify_all();
}

int clEventContainer::wait_for_signal() {
    // Wait for a signal to be ready
    std::unique_lock<std::mutex> lock(mux);
    while (!signal_set && !stopping) {
        cond_var.wait(lock);
    }

    // This signal hasn't been set, so we must be stopping without anything
    // to wait for.   If stopping is set, and there is a signal_set as well,
    // then we should still wait for the signal before exiting and
    // cleaning up the HSA memory space.
    if (!signal_set) {
        return -1;
    }

    // Then wait on the actual signal
    if (clWaitForEvents(1,&signal) != CL_SUCCESS) {
        ERROR("***** ERROR **** Unexpected event value **** ERROR **** ");
    }

    return 1;
}

void clEventContainer::wait_for_free_slot() {
    // Wait for signal_set == false, which implies that signal isn't set
    std::unique_lock<std::mutex> lock(mux);
    while (signal_set) {
        cond_var.wait(lock);
    }
}

void clEventContainer::stop() {
    {
        std::lock_guard<std::mutex> lock(mux);
        stopping = true;
    }
    cond_var.notify_all();
}
