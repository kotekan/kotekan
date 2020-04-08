#include "cudaEventContainer.hpp"

#include "errors.h"

#include <core/buffer.h>
#include <unistd.h>

void cudaEventContainer::set(void* sig) {
    signal = (cudaEvent_t)sig;
}

void* cudaEventContainer::get() {
    return signal;
}

void cudaEventContainer::unset() {
    signal = nullptr;
}

void cudaEventContainer::wait() {
    CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(signal));
}
