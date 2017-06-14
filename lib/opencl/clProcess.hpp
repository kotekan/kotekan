#ifndef GPU_THREAD_H
#define GPU_THREAD_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

#define SDK_SUCCESS 0

//check pagesize:
//getconf PAGESIZE
// result: 4096
#define PAGESIZE_MEM 4096

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

#include "pthread.h"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"

class clProcess : public KotekanProcess {
public:
    clProcess(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container);
    virtual ~clProcess();
    void main_thread();
    virtual void apply_config(uint64_t fpga_seq);

private:
    struct Buffer *in_buf;
    struct Buffer *out_buf;
    struct Buffer *beamforming_out_buf;
    struct Buffer *beamforming_out_incoh_buf;

    uint32_t gpu_id;

    // Config variables
    bool _use_beamforming;

};

void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void *data);

#endif