#include "cudaCommand.hpp"

#include "cudaStreamAssignment.hpp" // for resolve_cuda_stream, CUDA_STREAM_NEEDS_EXPLICIT
#include "cudaUtils.hpp"            // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h" // for cudaEventCreate, cudaEventDestroy, cudaEventRecord, cudaEv...
#include "visUtil.hpp"        // for StatTracker

#include "fmt.hpp" // for compile_string_to_view

#include <stdexcept> // for runtime_error
#include <utility>   // for pair

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

// default-constructed global so that people don't have to write
// "std::shared_ptr<cudaCommandState>()" in a bunch of places.  Basically a custom NULL.
std::shared_ptr<cudaCommandState> no_cuda_command_state;

cudaPipelineState::cudaPipelineState(int _gpu_frame_id) : gpu_frame_id(_gpu_frame_id) {}

cudaPipelineState::~cudaPipelineState() {}

void cudaPipelineState::set_flag(const std::string& key, bool val) {
    flags[key] = val;
}

bool cudaPipelineState::flag_exists(const std::string& key) const {
    // C++20
    // return flags.contains(key);
    return (flags.find(key) != flags.end());
}

bool cudaPipelineState::flag_is_set(const std::string& key) const {
    auto search = flags.find(key);
    if (search == flags.end())
        return false;
    return search->second;
}

void cudaPipelineState::set_int(const std::string& key, int64_t val) {
    intmap[key] = val;
}

int64_t cudaPipelineState::get_int(const std::string& key) const {
    return intmap.at(key);
}

cudaCommand::cudaCommand(Config& config_, const std::string& unique_name_,
                         bufferContainer& host_buffers_, cudaDeviceInterface& device_,
                         int instance_num_, std::shared_ptr<cudaCommandState> state_,
                         const std::string& default_kernel_command,
                         const std::string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, instance_num_, state_,
               default_kernel_command, default_kernel_file_name),
    start_event(nullptr), end_event(nullptr), device(device_) {
    _required_flag = config.get_default<std::string>(unique_name, "required_flag", "");
}

void cudaCommand::set_command_type(const gpuCommandType& type) {
    command_type = type;
    // An explicit `cuda_stream` is absolute and ignores the base.
    cuda_stream_id = config.get_default<int32_t>(unique_name, "cuda_stream", -1);
    if (cuda_stream_id >= 0)
        return;

    // Otherwise assign by role within the pipeline's triple: 3*base+0 copy-in, 3*base+1
    // copy-out, 3*base+2 kernel. The owning cudaProcess checks the result against its own
    // num_cuda_streams once every command is built (cudaProcess::collect_stream_ids).
    const int32_t base = config.get_default<int32_t>(unique_name, "cuda_stream_base", 0);
    cuda_stream_id = resolve_cuda_stream(command_type, base);
    if (cuda_stream_id == CUDA_STREAM_NEEDS_EXPLICIT)
        throw std::runtime_error("cuda_stream required for barrier type command object");
}

cudaCommand::~cudaCommand() {
    DEBUG("post_events Freed: {:s}", unique_name.c_str());
}

void cudaCommand::register_host_buffer(Buffer* buf) {
    if (instance_num != 0 || !buf->frame_size)
        return;
    for (int i = 0; i < buf->num_frames; i++) {
        uint flags;
        // only register the memory if it isn't already...
        if (cudaErrorInvalidValue == cudaHostGetFlags(&flags, buf->frames[i]))
            CHECK_CUDA_ERROR(cudaHostRegister(buf->frames[i], buf->frame_size, 0));
    }
}

void cudaCommand::unregister_host_buffer(Buffer* buf) {
    if (instance_num != 0 || !buf->frame_size)
        return;
    for (int i = 0; i < buf->num_frames; i++) {
        uint flags;
        // only unregister if it's actually been registered
        if (cudaSuccess == cudaHostGetFlags(&flags, buf->frames[i]))
            CHECK_CUDA_ERROR(cudaHostUnregister(buf->frames[i]));
    }
}

cudaEvent_t cudaCommand::execute_base(cudaPipelineState& pipestate,
                                      const std::vector<cudaEvent_t>& pre_events) {
    if (!should_execute(pipestate, pre_events))
        return nullptr;
    return execute(pipestate, pre_events);
}

bool cudaCommand::should_execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>&) {
    if (_required_flag.size() && !pipestate.flag_is_set(_required_flag)) {
        DEBUG("Required flag \"{:s}\" is not set; skipping stage", _required_flag);
        return false;
    }
    return true;
}

void cudaCommand::finalize_frame() {
    // An event that has not completed when finalize_frame runs means the frame was signalled
    // before its work finished, and the host frames it guards may already be released. Keep
    // this a hard error rather than tolerating cudaErrorNotReady: it is the only place that
    // early signal is visible.
    if (profiling && (start_event != nullptr) && (end_event != nullptr)) {
        float exec_time;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&exec_time, start_event, end_event));
        double active_time = exec_time * 1e-3; // convert ms to s
        excute_time->add_sample(active_time);
        utilization->add_sample(active_time / frame_arrival_period);
    } else {
        excute_time->add_sample(0.);
        utilization->add_sample(0.);
    }
    if (start_event != nullptr) {
        CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
        start_event = nullptr;
    }
    if (end_event != nullptr) {
        CHECK_CUDA_ERROR(cudaEventDestroy(end_event));
        end_event = nullptr;
    }
}

int32_t cudaCommand::get_cuda_stream_id() {
    return cuda_stream_id;
}

void cudaCommand::record_start_event() {
    if (profiling) {
        CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(start_event, device.getStream(cuda_stream_id)));
    }
}

cudaEvent_t cudaCommand::record_end_event() {
    CHECK_CUDA_ERROR(cudaEventCreate(&end_event));
    CHECK_CUDA_ERROR(cudaEventRecord(end_event, device.getStream(cuda_stream_id)));
    return end_event;
}
