#include "cudaCommand.hpp"

#include <cuda.h>
#include <nvrtc.h>

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
    } else {
        FATAL_ERROR("Null event in cudaCommand {:s}, this should never happen!", unique_name);
    }
}

void cudaCommand::build(const std::vector<std::string>& kernel_names,
                        std::vector<std::string>& opts) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvrtcResult res;

    DEBUG("Building! {:s}", kernel_command)

    // Load the kernel file contents into `program_buffer`
    fp = fopen(kernel_file_name.c_str(), "r");
    if (fp == NULL) {
        FATAL_ERROR("error loading file: {:s}", kernel_file_name.c_str());
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);

    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
    if (sizeRead < (int32_t)program_size)
        FATAL_ERROR("Error reading the file: {:s}", kernel_file_name);
    fclose(fp);

    // Create the program object
    nvrtcProgram prog;
    res = nvrtcCreateProgram(&prog, program_buffer, kernel_command.c_str(), 0, NULL, NULL);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        INFO("ERROR IN nvrtcCreateProgram: {}", error_str);
    }

    free(program_buffer);

    // Convert compiler options to a c-style array.
    std::vector<char*> cstrings;
    cstrings.reserve(opts.size());

    for (auto& str : opts)
        cstrings.push_back(&str[0]);

    // Compile the kernel
    res = nvrtcCompileProgram(prog, cstrings.size(), cstrings.data());
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcCompileProgram: {}", error_str);
        // Obtain compilation log from the program.
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        INFO("COMPILE LOG: {}", log);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    res = nvrtcGetPTX(prog, ptx);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcGetPTX: {}", error_str);
    }
    DEBUG2("PTX EXTRACTED");
    res = nvrtcDestroyProgram(&prog);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcDestroyProgram: {}", error_str);
    }

    CUresult err;
    CUmodule module;
    // Get the module with the kernels
    err = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        FATAL_ERROR("ERROR IN cuModuleLoadDataEx: {}", errStr);
    }

    for (auto& kernel_name : kernel_names) {
        runtime_kernels.emplace(kernel_name, nullptr);
        err = cuModuleGetFunction(&runtime_kernels[kernel_name], module, kernel_name.c_str());
        if (err != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(err, &errStr);
            FATAL_ERROR("ERROR IN cuModuleGetFunction for correlate: {}", errStr);
        }
    }
}
