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
#include "clDeviceInterface.hpp"
#include "clEventContainer.hpp"
#include "KotekanProcess.hpp"
#include "restServer.hpp"
#include "clCommand.hpp"
#include "json.hpp"

class clProcess : public KotekanProcess {
public:
    clProcess(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container);
    virtual ~clProcess();
    void main_thread() override;
    virtual void apply_config(uint64_t fpga_seq) override;
    void mem_reconcil_thread();
    void CL_CALLBACK results_thread();
    void profile_callback(connectionInstance& conn);
protected:
    vector<clEventContainer> final_signals;
    bufferContainer local_buffer_container;

    std::thread results_thread_handle;

    uint32_t _gpu_buffer_depth;

    std::thread mem_reconcil_thread_handle;

    uint32_t gpu_id;

    // Config variables
    bool _use_beamforming;

    clDeviceInterface * device;

    vector<clCommand *> commands;

};

void CL_CALLBACK read_complete(cl_event param_event, cl_int param_status, void *data);

#endif
