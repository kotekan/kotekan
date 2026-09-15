#include "cudaProcess.hpp"

#include "StageFactory.hpp"       // for REGISTER_KOTEKAN_STAGE
#include "cudaCommand.hpp"        // for cudaCommand, _factory_aliascudaCommandState, _factory_...
#include "cudaEventContainer.hpp" // for cudaEventContainer
#include "cudaUtils.hpp"          // for CHECK_CUDA_ERROR
#include "cuda_profiler_api.h"    // for cudaProfilerStart, cudaProfilerStop
#include "cuda_runtime_api.h"     // for cudaHostRegister, cudaStreamWaitEvent, cudaEventRecord...
#include "driver_types.h"         // for CUevent_st, cudaEvent_t, cudaEventDisableTiming, cudaH...
#include "factory.hpp"            // for FACTORY, FACTORY_VARIANT
#include "kotekanLogging.hpp"     // for DEBUG, DEBUG2

#include "fmt.hpp" // for compile_string_to_view

#include <mutex>    // for recursive_mutex, lock_guard
#include <stdint.h> // for uint32_t, int32_t

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

    // A pipeline that relies on a host-to-device copy overlapping a device-to-host one needs
    // two copy engines; with one they serialise and the overlap silently does not happen.
    // Opt-in (default 0, checked nowhere else) so that a test config depending on the overlap
    // fails loudly on hardware that cannot provide it rather than passing vacuously.
    const int32_t need_engines =
        config.get_default<int32_t>(unique_name, "require_async_engines", 0);
    if (need_engines > 0 && device->async_engine_count() < need_engines)
        FATAL_ERROR("{:s} requires {:d} async copy engines; GPU {:d} reports {:d}", unique_name,
                    need_engines, gpu_id, device->async_engine_count());

    CHECK_CUDA_ERROR(cudaProfilerStart());
    join_events.resize(_gpu_buffer_depth, nullptr);
    init();
}

cudaProcess::~cudaProcess() {
    // Ensure this thread is set to the correct device before final CUDA cleanup.
    if (device)
        device->set_thread_device();
    // Destroy the per-slot join events; nothing else owns them. Command events are freed on
    // results_thread instead, in finalize_frame, so this is the one frame-signal event freed
    // by the deleting thread -- which needs kotekanMode::join() to have run, as the commands
    // and final_signals ~gpuProcess frees already do.
    for (cudaEvent_t& e : join_events)
        if (e != nullptr)
            CHECK_CUDA_ERROR(cudaEventDestroy(e));
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
    cudaEvent_t final_event = nullptr;
    // The stream of the last command to return an event, and how many streams carry one. The
    // initial stream is never used: a frame with fewer than two carrying streams does not
    // join. A frame with NO events still signals nullptr, which cudaEventContainer hands to
    // cudaEventSynchronize and which aborts there -- unchanged by this commit, and a shape no
    // config in the tree currently produces.
    int32_t final_stream_id = 0;
    int streams_carrying = 0;

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
                if (events[command_stream_id] == nullptr)
                    streams_carrying++;
                events[command_stream_id] = event;
                final_event = event;
                final_stream_id = command_stream_id;
            }
        }

        // Streams are independent in-order queues and nothing crosses them unless a cudaSync*
        // command is configured to, so the last command to finish queuing is not necessarily
        // the last to finish -- while frame completion releases host frames, through the
        // cudaInputData and cudaOutputData overrides of finalize_frame.
        //
        // So the final command's stream waits on every other stream's last event of this
        // frame, stream 0 included, and the frame is signalled by an event recorded there,
        // where the chain already ends. One carrying stream cannot outlive itself, so the
        // common case skips all of it.
        if (streams_carrying > 1) {
            cudaStream_t join_stream = device->getStream(final_stream_id);
            for (size_t i = 0; i < events.size(); ++i)
                if (events[i] != nullptr && (int32_t)i != final_stream_id)
                    CHECK_CUDA_ERROR(cudaStreamWaitEvent(join_stream, events[i], 0));
            if (join_events[icommand] == nullptr)
                CHECK_CUDA_ERROR(
                    cudaEventCreateWithFlags(&join_events[icommand], cudaEventDisableTiming));
            CHECK_CUDA_ERROR(cudaEventRecord(join_events[icommand], join_stream));
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
