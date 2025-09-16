/**
 * @file
 * @brief Base class for defining CUDA commands to execute on GPUs
 *  - cudaEventContainer
 */

#ifndef CUDA_EVENT_CONTAINER_H
#define CUDA_EVENT_CONTAINER_H

#include "driver_types.h"        // for CUevent_st, cudaEvent_t
#include "gpuEventContainer.hpp" // for gpuEventContainer

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