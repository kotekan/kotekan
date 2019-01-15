#ifndef HCC_GPU_THREAD_H
#define HCC_GPU_THREAD_H

#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "buffer.h"

class hccGPUThread : public KotekanProcess {
public:
    hccGPUThread(Config &config,
                struct Buffer &in_buf,
                struct Buffer &out_buf,
                uint32_t gpu_id);
    virtual ~hccGPUThread();
    void main_thread() override;

private:
    struct Buffer &in_buf;
    struct Buffer &out_buf;

    uint32_t gpu_id;
    int * _zeros;
};

#endif