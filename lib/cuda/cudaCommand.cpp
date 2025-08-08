#include "cudaCommand.hpp"

#include <cuda.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>

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
    device(device_) {
    _required_flag = config.get_default<std::string>(unique_name, "required_flag", "");
    start_event = nullptr;
    end_event = nullptr;
}

void cudaCommand::set_command_type(const gpuCommandType& type) {
    command_type = type;
    // Use the cuda_stream if provided
    cuda_stream_id = config.get_default<int32_t>(unique_name, "cuda_stream", -1);

    if (cuda_stream_id >= device.get_num_streams()) {
        throw std::runtime_error(
            "Asked for a CUDA stream greater than the maximum number available");
    }
    // If the stream is set (not -1), we don't need to set a default below.
    if (cuda_stream_id >= 0)
        return;

    // If no stream set use a default stream, or generate an error
    switch (command_type) {
        case gpuCommandType::NOT_SET:
            throw std::runtime_error("No command type set");
            break;
        case gpuCommandType::COPY_IN:
            cuda_stream_id = 0;
            break;
        case gpuCommandType::COPY_OUT:
            cuda_stream_id = 1;
            break;
        case gpuCommandType::KERNEL:
            cuda_stream_id = 2;
            break;
        case gpuCommandType::BARRIER:
            throw std::runtime_error("cuda_stream required for barrier type command object");
            break;
        default:
            throw std::runtime_error("Invalid GPU Command type");
    }
}

cudaCommand::~cudaCommand() {
    DEBUG("post_events Freed: {:s}", unique_name.c_str());
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
    if (start_event)
        CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    start_event = nullptr;
    if (end_event != nullptr)
        CHECK_CUDA_ERROR(cudaEventDestroy(end_event));
    end_event = nullptr;
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
