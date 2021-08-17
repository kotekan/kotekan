#include "cudaCommand.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

cudaCommand::cudaCommand(Config& config_, const std::string& unique_name_,
                         bufferContainer& host_buffers_, cudaDeviceInterface& device_,
                         const std::string& default_kernel_command,
                         const std::string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, default_kernel_command,
               default_kernel_file_name),
    device(device_) {
    pre_events = (cudaEvent_t*)malloc(_gpu_buffer_depth * sizeof(cudaEvent_t));
    post_events = (cudaEvent_t*)malloc(_gpu_buffer_depth * sizeof(cudaEvent_t));
    for (int j = 0; j < _gpu_buffer_depth; ++j) {
        pre_events[j] = nullptr;
        post_events[j] = nullptr;
    }
}

cudaCommand::~cudaCommand() {
    free(pre_events);
    free(post_events);
    DEBUG("post_events Freed: %s", unique_name.c_str());
}

void cudaCommand::finalize_frame(int gpu_frame_id) {
    (void)gpu_frame_id;
    bool profiling = true;
    if (post_events[gpu_frame_id] != nullptr) {
        if (profiling) {
            float exec_time;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&exec_time, pre_events[gpu_frame_id],
                                                  post_events[gpu_frame_id]));
            double active_time = exec_time * 1e-3; // convert ms to s
            excute_time->add_sample(active_time);
            utilization->add_sample(active_time / frame_arrival_period);
        }
        CHECK_CUDA_ERROR(cudaEventDestroy(pre_events[gpu_frame_id]));
        pre_events[gpu_frame_id] = nullptr;
        CHECK_CUDA_ERROR(cudaEventDestroy(post_events[gpu_frame_id]));
        post_events[gpu_frame_id] = nullptr;
    } else
        ERROR("*** WTF? Null event!");
}
