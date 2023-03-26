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

gpuCommand* cudaProcess::create_command(const std::string& cmd_name,
                                        const std::string& unique_name) {
    auto cmd = FACTORY(cudaCommand)::create_bare(cmd_name, config, unique_name,
                                                 local_buffer_container, *device);
    DEBUG("Command added: {:s}", cmd_name.c_str());
    return cmd;
}

void cudaProcess::queue_commands(int gpu_frame_id) {
    std::vector<cudaEvent_t> events;
    events.resize(device->get_num_streams(), nullptr);
    int32_t command_stream_id = -1;

    bool quit = false;
    for (auto& command : commands) {
        // Feed the last signal into the next operation
        if (!quit) {
            command_stream_id = ((cudaCommand*)command)->get_cuda_stream_id();
            events[command_stream_id] = ((cudaCommand*)command)->execute(gpu_frame_id, events, &quit);
        } else
            ((cudaCommand*)command)->skipped_execute(gpu_frame_id, events);
    }
    // Wait on the very last event from the last command.
    // TODO, this should wait on the last event from every stream!
    final_signals[gpu_frame_id]->set_signal(events[command_stream_id]);
    DEBUG2("Commands executed.");
}

void cudaProcess::register_host_memory(struct Buffer* host_buffer) {
    // Register the host memory in in_buf with the OpenCL run time.
    for (int i = 0; i < host_buffer->num_frames; i++) {
        cudaHostRegister(host_buffer->frames[i], host_buffer->aligned_frame_size,
                         cudaHostRegisterDefault);
        DEBUG("Registered frame: {:s}[{:d}]", host_buffer->buffer_name, i);
    }
}
