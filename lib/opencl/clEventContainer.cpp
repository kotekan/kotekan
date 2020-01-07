#include "clEventContainer.hpp"

#include "kotekanLogging.hpp"

#include <unistd.h>

void clEventContainer::set(void* sig) {
    signal = (cl_event)sig;
}

void* clEventContainer::get() {
    return signal;
}

void clEventContainer::unset() {
    signal = nullptr;
}

void clEventContainer::wait() {
    if (clWaitForEvents(1, &signal) != CL_SUCCESS) {
        FATAL_ERROR_NON_OO("***** ERROR **** Unexpected event value **** ERROR **** ");
    }
}
