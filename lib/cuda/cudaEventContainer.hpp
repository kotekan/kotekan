/**
 * @file cudaEventContainer.h
 * @brief Base class for defining CUDA commands to execute on GPUs
 *  - cudaEventContainer
 */

#ifndef CUDA_EVENT_CONTAINER_H
#define CUDA_EVENT_CONTAINER_H

#include "gpuEventContainer.hpp"

#include <condition_variable>
#include <mutex>
#include <signal.h>
#include <thread>

#include "cuda_runtime_api.h"
#include "cudaUtils.hpp"

/**
 * @class cudaEventContainer
 * @brief Class to handle CUDA events for pipelining kernels & copies.
 *
 * @author Keith Vanderlinde
 */

class cudaEventContainer final : public gpuEventContainer {

public:
    void set(void* sig) override;
    void* get() override;
    void unset() override;
    void wait() override;

private:
    cudaEvent_t signal;
};

#endif // CUDA_EVENT_CONTAINER_H