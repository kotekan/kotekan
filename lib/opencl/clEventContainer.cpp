#include "clEventContainer.hpp"

#include "errors.h"

#include <unistd.h>

void clEventContainer::set(void* sig) {
    signal = (cl_event)sig;
}

void* clEventContainer::get() {
    return signal;
}

void clEventContainer::unset() {
    signal = NULL;
}

void clEventContainer::wait() {
    if (clWaitForEvents(1, &signal) != CL_SUCCESS) {
        FATAL_ERROR("***** ERROR **** Unexpected event value **** ERROR **** ");
    }
}
