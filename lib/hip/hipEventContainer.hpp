/**
 * @file
 * @brief Base class for defining HIP commands to execute on GPUs
 *  - hipEventContainer
 */

#ifndef HIP_EVENT_CONTAINER_H
#define HIP_EVENT_CONTAINER_H

#include "gpuEventContainer.hpp"
#include "hip/hip_runtime_api.h"
#include "hipUtils.hpp"

#include <condition_variable>
#include <mutex>
#include <signal.h>
#include <thread>

/**
 * @class hipEventContainer
 * @brief Class to handle HIP events for pipelining kernels & copies.
 *
 * @author Andre Renard
 */
class hipEventContainer final : public gpuEventContainer {

public:
    void set(void* sig) override;
    void* get() override;
    void unset() override;
    void wait() override;

private:
    hipEvent_t signal;
};

#endif // HIP_EVENT_CONTAINER_H