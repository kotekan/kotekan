#include "hipProcess.hpp"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "factory.hpp"

//#include "hip/hip_profiler_api.h"
#include "unistd.h"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using kotekan::bufferContainer;
using kotekan::Config;

using namespace std;

REGISTER_KOTEKAN_STAGE(hipProcess);

// TODO Remove the GPU_ID from this constructor
hipProcess::hipProcess(Config& config_, const string& unique_name,
                       bufferContainer& buffer_container) :
    gpuProcess(config_, unique_name, buffer_container) {
    device = new hipDeviceInterface(config_, gpu_id, _gpu_buffer_depth);
    dev = device;
    device->prepareStreams();
    // CHECK_HIP_ERROR(hipProfilerStart());
    init();
}

hipProcess::~hipProcess() {
    // CHECK_HIP_ERROR(hipProfilerStop());
}

gpuEventContainer* hipProcess::create_signal() {
    return new hipEventContainer();
}

gpuCommand* hipProcess::create_command(const std::string& cmd_name,
                                       const std::string& unique_name) {
    auto cmd = FACTORY(hipCommand)::create_bare(cmd_name, config, unique_name,
                                                local_buffer_container, *device);
    DEBUG("Command added: {:s}", cmd_name.c_str());
    return cmd;
}

void hipProcess::queue_commands(int gpu_frame_id) {
    hipEvent_t signal = nullptr;
    for (auto& command : commands) {
        // Feed the last signal into the next operation
        signal = ((hipCommand*)command)->execute(gpu_frame_id, signal);
    }
    final_signals[gpu_frame_id]->set_signal(signal);
    INFO("Commands executed.");
}

void hipProcess::register_host_memory(struct Buffer* host_buffer) {
    // Register the host memory in in_buf with the OpenCL run time.
    for (int i = 0; i < host_buffer->num_frames; i++) {
        hipHostRegister(host_buffer->frames[i], host_buffer->aligned_frame_size,
                        hipHostRegisterDefault);
        DEBUG("Registered frame: {:s}[{:d}]", host_buffer->buffer_name, i);
    }
}