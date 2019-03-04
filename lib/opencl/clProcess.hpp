#ifndef CL_PROCESS_H
#define CL_PROCESS_H

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b)&0x0F)

#include "clCommand.hpp"
#include "clDeviceInterface.hpp"
#include "clEventContainer.hpp"
#include "gpuProcess.hpp"

class clProcess final : public gpuProcess {
public:
    clProcess(kotekan::Config& config, const string& unique_name,
              kotekan::bufferContainer& buffer_container);
    virtual ~clProcess();

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
    double frame_arrival_period;

    // Config variables
    bool _use_beamforming;

    clDeviceInterface * device;

    vector<clCommand *> commands;

    clDeviceInterface* device;
};

#endif // CL_PROCESS_H
