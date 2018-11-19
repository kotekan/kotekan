#ifndef CL_EVENT_CONTAINER_H
#define CL_EVENT_CONTAINER_H

#include <condition_variable>
#include <mutex>
#include <thread>
#include <signal.h>
#include "gpuEventContainer.hpp"

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

class clEventContainer final : public gpuEventContainer {

public:
    void set(void *sig) override;
    void *get() override;
    void unset() override;
    void wait() override;
private:
    cl_event signal;
};

#endif //CL_EVENT_CONTAINER_H