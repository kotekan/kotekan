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
/*
    if (kernel_command != "") {
        CHECK_CL_ERROR(clReleaseKernel(kernel));
        DEBUG("kernel Freed");
        CHECK_CL_ERROR(clReleaseProgram(program));
        DEBUG("program Freed");
    }
    */
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
            last_gpu_execution_time = exec_time * 1e-6; //concert ms to ns
        }
        cudaEventDestroy(pre_events[gpu_frame_id]);
        pre_events[gpu_frame_id] = NULL;
        cudaEventDestroy(post_events[gpu_frame_id]);
        post_events[gpu_frame_id] = NULL;
    } else
        ERROR("*** WTF? Null event!");
}


// Specialist functions:
void cudaCommand::build() {
/*
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    cl_int err;

    if (kernel_command != "") {
        DEBUG2("Building! %s", kernel_command.c_str())
        fp = fopen(kernel_file_name.c_str(), "r");
        if (fp == NULL) {
            ERROR("error loading file: %s", kernel_file_name.c_str());
            raise(SIGINT);
        }
        fseek(fp, 0, SEEK_END);
        program_size = ftell(fp);
        rewind(fp);

        program_buffer = (char*)malloc(program_size + 1);
        program_buffer[program_size] = '\0';
        int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
        if (sizeRead < (int32_t)program_size)
            ERROR("Error reading the file: %s", kernel_file_name.c_str());
        fclose(fp);
        program =
            clCreateProgramWithSource(((clDeviceInterface*)&device)->get_context(), (cl_uint)1,
                                      (const char**)&program_buffer, &program_size, &err);
        CHECK_CL_ERROR(err);

        program_size = 0;
        free(program_buffer);
        DEBUG2("Built! %s", kernel_command.c_str())
    }
    */
}
