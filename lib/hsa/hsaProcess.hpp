#ifndef GPU_HSA_THREAD_H
#define GPU_HSA_THREAD_H

#include <condition_variable>
#include <mutex>
#include <thread>

#include "json.hpp"
#include "fpga_header_functions.h"
#include "KotekanProcess.hpp"
#include "bufferContainer.hpp"
#include "hsaDeviceInterface.hpp"
#include "hsaCommand.hpp"
#include "signalContainer.hpp"
#include "bufferContainer.hpp"
#include "restServer.hpp"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

using nlohmann::json;

class hsaProcess : public KotekanProcess {
public:
    hsaProcess(Config& config, const string& unique_name,
              bufferContainer &buffer_container);
    virtual ~hsaProcess();

    void main_thread();

    void results_thread();

    virtual void apply_config(uint64_t fpga_seq);

    void profile_callback(connectionInstance& conn);

private:

    vector<signalContainer> final_signals;

//    hsaCommandFactory * factory;
    hsaDeviceInterface * device;

    std::thread results_thread_handle;

    uint32_t _gpu_buffer_depth;

    uint32_t gpu_id;

    bool log_profiling;

    // TODO should this be removed?
    bufferContainer local_buffer_container;

    // The mean expected time between frames in seconds
    double frame_arrival_period;

    std::string endpoint;


    vector<hsaCommand *> commands;
};

#endif
