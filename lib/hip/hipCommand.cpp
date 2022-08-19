#include "hipCommand.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

hipCommand::hipCommand(Config& config_, const string& unique_name_, bufferContainer& host_buffers_,
                       hipDeviceInterface& device_, const string& default_kernel_command,
                       const string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, default_kernel_command,
               default_kernel_file_name),
    device(device_) {
    pre_events = (hipEvent_t*)malloc(_gpu_buffer_depth * sizeof(hipEvent_t));
    post_events = (hipEvent_t*)malloc(_gpu_buffer_depth * sizeof(hipEvent_t));
    for (int j = 0; j < _gpu_buffer_depth; ++j) {
        pre_events[j] = NULL;
        post_events[j] = NULL;
    }
}

hipCommand::~hipCommand() {
    free(pre_events);
    free(post_events);
    DEBUG("post_events Freed: %s", unique_name.c_str());
}

void hipCommand::finalize_frame(int gpu_frame_id) {
    (void)gpu_frame_id;
    bool profiling = true;
    if (post_events[gpu_frame_id] != NULL) {
        if (profiling) {
            float exec_time;
            CHECK_HIP_ERROR(hipEventElapsedTime(&exec_time, pre_events[gpu_frame_id],
                                                post_events[gpu_frame_id]));
            double active_time = exec_time * 1e-3; // convert ms to s
            excute_time->add_sample(active_time);
            utilization->add_sample(active_time / frame_arrival_period);
        }
        CHECK_HIP_ERROR(hipEventDestroy(pre_events[gpu_frame_id]));
        pre_events[gpu_frame_id] = NULL;
        CHECK_HIP_ERROR(hipEventDestroy(post_events[gpu_frame_id]));
        post_events[gpu_frame_id] = NULL;
    } else
        ERROR("*** WTF? Null event!");
}
