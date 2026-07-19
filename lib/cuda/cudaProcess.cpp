#include "cudaProcess.hpp"

#include <stdint.h>                // for uint32_t, int32_t
#include <mutex>                   // for recursive_mutex, lock_guard

#include "StageFactory.hpp"        // for REGISTER_KOTEKAN_STAGE
#include "cudaCommand.hpp"         // for cudaCommand, _factory_aliascudaCommandState, _factory_...
#include "cudaEventContainer.hpp"  // for cudaEventContainer
#include "cudaUtils.hpp"           // for CHECK_CUDA_ERROR
#include "cuda_profiler_api.h"     // for cudaProfilerStart, cudaProfilerStop
#include "cuda_runtime_api.h"      // for cudaHostRegister
#include "driver_types.h"          // for CUevent_st, cudaEvent_t, cudaHostRegisterDefault
#include "factory.hpp"             // for FACTORY, FACTORY_VARIANT
#include "kotekanLogging.hpp"      // for DEBUG, DEBUG2
#include "fmt.hpp"                 // for compile_string_to_view

using kotekan::bufferContainer;
using kotekan::Config;

using namespace std;

REGISTER_KOTEKAN_STAGE(cudaProcess);

// TODO Remove the GPU_ID from this constructor
cudaProcess::cudaProcess(Config& config_, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    gpuProcess(config_, unique_name, buffer_container) {
    std::string device_name =
        config_.get_default<std::string>(unique_name, "device", "device_" + std::to_string(gpu_id));
    device = cudaDeviceInterface::get(gpu_id, device_name, config_);
    dev = device.get();
    // Tell the Cuda runtime to associate this gpu_id with this thread/Stage.
    device->set_thread_device();

    uint32_t num_streams = config.get_default<uint32_t>(unique_name, "num_cuda_streams", 3);

    device->prepareStreams(num_streams);
    CHECK_CUDA_ERROR(cudaProfilerStart());
    init();
}

cudaProcess::~cudaProcess() {
    // Ensure this thread is set to the correct device before final CUDA cleanup.
    if (device)
        device->set_thread_device();
    CHECK_CUDA_ERROR(cudaProfilerStop());
    // With weak_ptr cache, simply letting the last shared_ptr go triggers cleanup.
}

gpuEventContainer* cudaProcess::create_signal() {
    return new cudaEventContainer();
}

std::vector<gpuCommand*> cudaProcess::create_command(const std::string& cmd_name,
                                                     const std::string& unique_name) {
    std::vector<gpuCommand*> cmds;
    // Create the cudaCommandState object, if used, for this command class.
    std::shared_ptr<cudaCommandState> st;
    if (FACTORY(cudaCommandState)::exists(cmd_name))
        st = FACTORY(cudaCommandState)::create_shared(cmd_name, config, unique_name,
                                                      local_buffer_container, *device);
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

void cudaProcess::queue_commands(int gpu_frame_counter) {
    std::vector<cudaEvent_t> events;
    events.resize(device->get_num_streams(), nullptr);
    if (join_events.size() < (size_t)_gpu_buffer_depth)
        join_events.resize(_gpu_buffer_depth, nullptr);
    cudaEvent_t final_event = nullptr;

    int icommand = gpu_frame_counter % _gpu_buffer_depth;
    {
        // Grab the lock for queuing GPU commands
        std::lock_guard<std::recursive_mutex> lock(device->gpu_command_mutex);

        // Create the state object that will get passed through this pipeline
        cudaPipelineState pipestate(gpu_frame_counter);

        for (auto& command : commands) {
            // Feed the last signal into the next operation
            cudaEvent_t event = ((cudaCommand*)command[icommand])->execute_base(pipestate, events);
            if (event != nullptr) {
                int32_t command_stream_id = ((cudaCommand*)command[icommand])->get_cuda_stream_id();
                events[command_stream_id] = event;
                final_event = event;
            }
        }
    }
    // Frame completion must wait on the last event from EVERY stream (the old
    // last-command-only signal was the long-standing TODO here): with parallel
    // per-stream command chains (e.g. the unified L1 tracker, three cudaGnssTrack+
    // output pairs on streams 1-3), the final command finishes first on ITS stream
    // while siblings are still running -- downstream then consumed half-written
    // frames (measured 2026-07-19: gal/bds epl frames with n_prn 0). Join: make
    // stream 0 wait each stream's last event, record one event there, signal that.
    {
        cudaStream_t s0 = device->getStream(0);
        bool multi = false;
        for (size_t i = 1; i < events.size(); ++i)
            if (events[i] != nullptr && events[i] != final_event)
                multi = true;
        if (multi) {
            for (size_t i = 0; i < events.size(); ++i)
                if (events[i] != nullptr)
                    CHECK_CUDA_ERROR(cudaStreamWaitEvent(s0, events[i], 0));
            if (join_events[icommand] != nullptr)
                CHECK_CUDA_ERROR(cudaEventDestroy(join_events[icommand]));
            CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&join_events[icommand],
                                                      cudaEventDisableTiming));
            CHECK_CUDA_ERROR(cudaEventRecord(join_events[icommand], s0));
            final_event = join_events[icommand];
        }
    }
    final_signals[icommand]->set_signal(final_event);
    DEBUG2("Commands executed.");
}

void cudaProcess::register_host_memory(Buffer* host_buffer) {
    // Register the host memory in buffers with the Cuda run time.
    for (int i = 0; i < host_buffer->num_frames; i++) {
        cudaHostRegister(host_buffer->frames[i], host_buffer->aligned_frame_size,
                         cudaHostRegisterDefault);
        DEBUG("Registered frame: {:s}[{:d}]", host_buffer->buffer_name, i);
    }
}
