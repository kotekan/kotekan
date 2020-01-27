#include "cudaProcess.hpp"

#include "cuda_profiler_api.h"
#include "unistd.h"
#include "util.h"

#include <iostream>
#include <sys/time.h>

using kotekan::bufferContainer;
using kotekan::Config;

using namespace std;

REGISTER_KOTEKAN_STAGE(cudaProcess);

// TODO Remove the GPU_ID from this constructor
cudaProcess::cudaProcess(Config& config_, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    gpuProcess(config_, unique_name, buffer_container) {
    device = new cudaDeviceInterface(config_, gpu_id, _gpu_buffer_depth);
    dev = device;
    device->prepareStreams();
    CHECK_CUDA_ERROR(cudaProfilerStart());
    init();
}

cudaProcess::~cudaProcess() {
    CHECK_CUDA_ERROR(cudaProfilerStop());
}

gpuEventContainer* cudaProcess::create_signal() {
    return new cudaEventContainer();
}

gpuCommand* cudaProcess::create_command(const std::string& cmd_name,
                                        const std::string& unique_name) {
    auto cmd = FACTORY(cudaCommand)::create_bare(cmd_name, config, unique_name,
                                                 local_buffer_container, *device);
    DEBUG("Command added: {:s}", cmd_name.c_str());
    return cmd;
}

void cudaProcess::queue_commands(int gpu_frame_id) {
    cudaEvent_t signal = nullptr;
    for (auto& command : commands) {
        // Feed the last signal into the next operation
        signal = ((cudaCommand*)command)->execute(gpu_frame_id, signal);
    }
    final_signals[gpu_frame_id]->set_signal(signal);
    INFO("Commands executed.");
}
