#ifndef GPU_HSA_THREAD_H
#define GPU_HSA_THREAD_H

#include <condition_variable>
#include <mutex>
#include <thread>

#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "bufferContainer.hpp"
#include "hsaDeviceInterface.hpp"
#include "hsaCommandFactory.hpp"
#include "hsaCommand.hpp"
#include "signalContainer.hpp"
#include "bufferContainer.hpp"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

class hsaThread : public KotekanProcess {
public:
    hsaThread(Config& config, const string& unique_name,
              bufferContainer &buffer_container);
    virtual ~hsaThread();

    void main_thread();

    void results_thread();

    virtual void apply_config(uint64_t fpga_seq);

private:

    vector<signalContainer> final_signals;

    hsaCommandFactory * factory;
    hsaDeviceInterface * device;

    std::thread results_thread_handle;

    uint32_t _gpu_buffer_depth;

    uint32_t gpu_id;

    // TODO should this be removed?
    bufferContainer local_buffer_container;
};

#endif