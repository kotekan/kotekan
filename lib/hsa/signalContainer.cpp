#include "signalContainer.hpp"
#include "errors.h"
#include <unistd.h>


signalContainer::signalContainer() {
    reset();
    stopping = false;
}

signalContainer::signalContainer(const signalContainer& obj) {
    // FIXME
    (void)obj;
}

signalContainer::~signalContainer() {
}

void signalContainer::reset() {
    {
        std::lock_guard<std::mutex> lock(mux);
        signal_set = false;
        signal.handle = 0;
    }
    cond_var.notify_all();
}

void signalContainer::set_signal(hsa_signal_t sig) {
    {
        std::lock_guard<std::mutex> lock(mux);
        signal = sig;
        signal_set = true;
    }
    cond_var.notify_all();
}

int signalContainer::wait_for_signal() {
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
    if (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0) {
        ERROR("***** ERROR **** Unexpected signal value **** ERROR **** ");
    }

    return 1;
}

void signalContainer::wait_for_free_slot() {
    // Wait for signal_set == false, which implies that signal isn't set
    std::unique_lock<std::mutex> lock(mux);
    while (signal_set) {
        cond_var.wait(lock);
    }
}

void signalContainer::stop() {
    {
        std::lock_guard<std::mutex> lock(mux);
        stopping = true;
    }
    cond_var.notify_all();
}
