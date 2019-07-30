#include "cudaCommand.hpp"

#include <iostream>

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

cudaCommand::cudaCommand(Config& config_, const string& unique_name_, bufferContainer& host_buffers_,
                      cudaDeviceInterface& device_,
                      const string& default_kernel_command, const string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, default_kernel_command,
               default_kernel_file_name),
    device(device_) {
    pre_events = (cudaEvent_t*)malloc(_gpu_buffer_depth * sizeof(cudaEvent_t));
    post_events = (cudaEvent_t*)malloc(_gpu_buffer_depth * sizeof(cudaEvent_t));
    for (int j = 0; j < _gpu_buffer_depth; ++j){
        pre_events[j] = NULL;
        post_events[j] = NULL;
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
    if (post_events[gpu_frame_id] != NULL) {
        if (profiling) {
            float exec_time;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&exec_time,
                                                  pre_events[gpu_frame_id],
                                                  post_events[gpu_frame_id]));
            last_gpu_execution_time = exec_time * 1e-3; //concert ms to s
        }
        CHECK_CUDA_ERROR(cudaEventDestroy(pre_events[gpu_frame_id]));
        pre_events[gpu_frame_id] = NULL;
        CHECK_CUDA_ERROR(cudaEventDestroy(post_events[gpu_frame_id]));
        post_events[gpu_frame_id] = NULL;
    } else
        ERROR("*** WTF? Null event!");
}

