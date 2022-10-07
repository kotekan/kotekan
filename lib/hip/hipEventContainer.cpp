#include "hipEventContainer.hpp"

#include "errors.h"

#include <unistd.h>

void hipEventContainer::set(void* sig) {
    signal = (hipEvent_t)sig;
}

void* hipEventContainer::get() {
    return signal;
}

void hipEventContainer::unset() {
    signal = NULL;
}

void hipEventContainer::wait() {
    CHECK_HIP_ERROR_NON_OO(hipEventSynchronize(signal));
}
