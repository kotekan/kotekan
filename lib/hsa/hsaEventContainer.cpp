#include "hsaEventContainer.hpp"

#include "errors.h"
#include "kotekanLogging.hpp"

#include <unistd.h>


void hsaEventContainer::set(void* sig) {
    signal = *(hsa_signal_t*)sig;
}

void* hsaEventContainer::get() {
    return &signal;
}

void hsaEventContainer::unset() {
    // signal = 0;//*(hsa_signal_t*)NULL;
}

void hsaEventContainer::wait() {
    // Then wait on the actual signal
    if (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                  HSA_WAIT_STATE_BLOCKED)
        != 0) {
        ERROR_NON_OO("***** ERROR **** Unexpected signal value **** ERROR **** ");
    }
}
