#include "clCommand.hpp"

#include <iostream>

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

clCommand::clCommand(Config& config_, const std::string& unique_name_,
                     bufferContainer& host_buffers_, clDeviceInterface& device_,
                     const std::string& default_kernel_command,
                     const std::string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, default_kernel_command,
               default_kernel_file_name),
    device(device_) {
    post_events = (cl_event*)malloc(_gpu_buffer_depth * sizeof(cl_event));
    for (int j = 0; j < _gpu_buffer_depth; ++j)
        post_events[j] = nullptr;
}

clCommand::~clCommand() {
    if (kernel_command != "") {
        CHECK_CL_ERROR(clReleaseKernel(kernel));
        DEBUG("kernel Freed");
        CHECK_CL_ERROR(clReleaseProgram(program));
        DEBUG("program Freed");
    }
    free(post_events);
    DEBUG("post_events Freed: {:s}", unique_name);
}

void clCommand::finalize_frame(int gpu_frame_id) {
    bool profiling = true;
    if (post_events[gpu_frame_id] != nullptr) {
        if (profiling) {
            cl_ulong start_time, stop_time;
            CHECK_CL_ERROR(clGetEventProfilingInfo(post_events[gpu_frame_id],
                                                   CL_PROFILING_COMMAND_START, sizeof(start_time),
                                                   &start_time, nullptr));
            CHECK_CL_ERROR(clGetEventProfilingInfo(post_events[gpu_frame_id],
                                                   CL_PROFILING_COMMAND_END, sizeof(stop_time),
                                                   &stop_time, nullptr));
            last_gpu_execution_time = ((double)(stop_time - start_time)) * 1e-9;
        }

        CHECK_CL_ERROR(clReleaseEvent(post_events[gpu_frame_id]));
        post_events[gpu_frame_id] = nullptr;
    } else
        FATAL_ERROR("Null OpenCL event!");
}


// Specialist functions:
void clCommand::build() {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    cl_int err;

    if (kernel_command != "") {
        DEBUG2("Building! {:s}", kernel_command)
        fp = fopen(kernel_file_name.c_str(), "r");
        if (fp == nullptr) {
            FATAL_ERROR("error loading file: {:s}", kernel_file_name);
        }
        fseek(fp, 0, SEEK_END);
        program_size = ftell(fp);
        rewind(fp);

        program_buffer = (char*)malloc(program_size + 1);
        program_buffer[program_size] = '\0';
        int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
        if (sizeRead < (int32_t)program_size)
            ERROR("Error reading the file: {:s}", kernel_file_name);
        fclose(fp);
        program =
            clCreateProgramWithSource(((clDeviceInterface*)&device)->get_context(), (cl_uint)1,
                                      (const char**)&program_buffer, &program_size, &err);
        CHECK_CL_ERROR(err);

        program_size = 0;
        free(program_buffer);
        DEBUG2("Built! {:s}", kernel_command)
    }
}

void clCommand::setKernelArg(cl_uint param_ArgPos, cl_mem param_Buffer) {
    CHECK_CL_ERROR(clSetKernelArg(kernel, param_ArgPos, sizeof(void*), (void*)&param_Buffer));
}
