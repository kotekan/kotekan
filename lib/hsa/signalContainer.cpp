#include "signalContainer.hpp"

signalContainer::signalContainer() {
    reset();
}

signalContainer::signalContainer(const signalContainer& obj) {
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

void signalContainer::wait_for_signal() {
    // Wait for a signal to be ready
    std::unique_lock<std::mutex> lock(mux);
    while (!signal_set) {
        cond_var.wait(lock);
    }

    // Then wait on the actual signal
    while ( hsa_signal_wait_acquire(signal,
                                    HSA_SIGNAL_CONDITION_LT, 1,
                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED) > 0 );
}

void signalContainer::wait_for_free_slot() {
    // Wait for signal_set == false, which implies that signal isn't set
    std::unique_lock<std::mutex> lock(mux);
    while (signal_set) {
        cond_var.wait(lock);
    }
}
