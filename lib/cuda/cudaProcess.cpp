#include "cudaProcess.hpp"

#include "Stage.hpp"
#include "StageFactory.hpp"
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
    device = new cudaDeviceInterface(config_, unique_name, gpu_id, _gpu_buffer_depth);
    dev = device;

    uint32_t num_streams = config.get_default<uint32_t>(unique_name, "num_cuda_streams", 3);

    device->prepareStreams(num_streams);
    CHECK_CUDA_ERROR(cudaProfilerStart());
    init();
}

cudaProcess::~cudaProcess() {
    CHECK_CUDA_ERROR(cudaProfilerStop());
}

gpuEventContainer* cudaProcess::create_signal() {
    return new cudaEventContainer();
}

std::vector<gpuCommand*> cudaProcess::create_command(const std::string& cmd_name,
                                                     const std::string& unique_name) {
    std::vector<gpuCommand*> cmds;
    // Create the cudaCommandState object, if used, for this command class.
    std::shared_ptr<cudaCommandState> st = FACTORY(cudaCommandState)::create_shared_if_exists(
        cmd_name, config, unique_name, local_buffer_container, *device);
    for (uint32_t i = 0; i < _gpu_buffer_depth; i++) {
        gpuCommand* cmd;
        if (st)
            // Create the cudaCommand object (with state arg)
            cmd = FACTORY_VARIANT(state, cudaCommand)::create_bare(
                cmd_name, config, unique_name, local_buffer_container, *device, i, st);
        else
            // Create the cudaCommand object (without state arg)
            cmd = FACTORY(cudaCommand)::create_bare(cmd_name, config, unique_name,
                                                    local_buffer_container, *device, i);
        cmds.push_back(cmd);
    }
    DEBUG("Command added: {:s}", cmd_name.c_str());
    return cmds;
}

void cudaProcess::queue_commands(int gpu_frame_id, int gpu_frame_counter) {
    std::vector<cudaEvent_t> events;
    events.resize(device->get_num_streams(), nullptr);

    cudaPipelineState pipestate(gpu_frame_id);
    pipestate.set_int("gpu_frame_counter", gpu_frame_counter);
    cudaEvent_t final_event = nullptr;

    for (auto& command : commands) {
        // Feed the last signal into the next operation
        cudaEvent_t event = ((cudaCommand*)command[gpu_frame_id])->execute_base(pipestate, events);
        if (event != nullptr) {
            int32_t command_stream_id = ((cudaCommand*)command[gpu_frame_id])->get_cuda_stream_id();
            events[command_stream_id] = event;
            final_event = event;
        }
    }
    // Wait on the very last event from the last command.
    // TODO, this should wait on the last event from every stream!
    final_signals[gpu_frame_id]->set_signal(final_event);
    DEBUG2("Commands executed.");
}

void cudaProcess::register_host_memory(Buffer* host_buffer) {
    // Register the host memory in in_buf with the OpenCL run time.
    for (int i = 0; i < host_buffer->num_frames; i++) {
        cudaHostRegister(host_buffer->frames[i], host_buffer->aligned_frame_size,
                         cudaHostRegisterDefault);
        DEBUG("Registered frame: {:s}[{:d}]", host_buffer->buffer_name, i);
    }
}
