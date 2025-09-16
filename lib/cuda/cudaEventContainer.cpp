#include "cudaEventContainer.hpp"

#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR_NON_OO
#include "cuda_runtime_api.h" // for cudaEventSynchronize

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
