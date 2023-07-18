#include "cudaCommand.hpp"

#include <cuda.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>

using kotekan::bufferContainer;
using kotekan::Config;

using std::string;
using std::to_string;

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
                         const std::string& default_kernel_command,
                         const std::string& default_kernel_file_name) :
    gpuCommand(config_, unique_name_, host_buffers_, device_, default_kernel_command,
               default_kernel_file_name),
    device(device_) {
    _required_flag = config.get_default<std::string>(unique_name, "required_flag", "");
    start_events = (cudaEvent_t*)malloc(_gpu_buffer_depth * sizeof(cudaEvent_t));
    end_events = (cudaEvent_t*)malloc(_gpu_buffer_depth * sizeof(cudaEvent_t));
    for (int j = 0; j < _gpu_buffer_depth; ++j) {
        start_events[j] = nullptr;
        end_events[j] = nullptr;
    }
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
    free(start_events);
    free(end_events);
    DEBUG("post_events Freed: {:s}", unique_name.c_str());
}

cudaEvent_t cudaCommand::execute_base(cudaPipelineState& pipestate,
                                      const std::vector<cudaEvent_t>& pre_events) {
    if (_required_flag.size() && !pipestate.flag_is_set(_required_flag)) {
        DEBUG("Required flag \"{:s}\" is not set; skipping stage", _required_flag);
        return nullptr;
    }
    return execute(pipestate, pre_events);
}

void cudaCommand::finalize_frame(int gpu_frame_id) {
    if (profiling && (start_events[gpu_frame_id] != nullptr)
        && (end_events[gpu_frame_id] != nullptr)) {
        float exec_time;
        CHECK_CUDA_ERROR(
            cudaEventElapsedTime(&exec_time, start_events[gpu_frame_id], end_events[gpu_frame_id]));
        double active_time = exec_time * 1e-3; // convert ms to s
        excute_time->add_sample(active_time);
        utilization->add_sample(active_time / frame_arrival_period);
    } else {
        excute_time->add_sample(0.);
        utilization->add_sample(0.);
    }
    if (start_events[gpu_frame_id])
        CHECK_CUDA_ERROR(cudaEventDestroy(start_events[gpu_frame_id]));
    start_events[gpu_frame_id] = nullptr;
    if (end_events[gpu_frame_id] != nullptr)
        CHECK_CUDA_ERROR(cudaEventDestroy(end_events[gpu_frame_id]));
    end_events[gpu_frame_id] = nullptr;
}

int32_t cudaCommand::get_cuda_stream_id() {
    return cuda_stream_id;
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
    if (fp == nullptr) {
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
    res = nvrtcCreateProgram(&prog, program_buffer, kernel_command.c_str(), 0, nullptr, nullptr);
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
    err = cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
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
        if (runtime_kernels[kernel_name] == nullptr) {
            FATAL_ERROR("Failed to find kernel name \"{}\" in compiled PTX module", kernel_name);
        }
    }
}

void cudaCommand::build_ptx(const std::vector<std::string>& kernel_names,
                            const std::vector<std::string>& opts) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvPTXCompileResult nv_res;
    CUresult cu_res;
    nvPTXCompilerHandle compiler = nullptr;
    size_t elf_size;
    char* elf;
    CUmodule module;

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

    // Create the compiler
    nv_res = nvPTXCompilerCreate(&compiler, program_size, program_buffer);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        // TODO Report ENUM names.
        FATAL_ERROR("Could not create PTX compiler, error code: {:d}", nv_res);
        return;
    }

    // Convert compiler options to a c-style array.
    std::vector<const char*> cstring_opts;
    cstring_opts.reserve(opts.size());
    for (auto& str : opts)
        cstring_opts.push_back(str.c_str());

    // Compile the code
    nv_res = nvPTXCompilerCompile(compiler, cstring_opts.size(), cstring_opts.data());
    // TODO Abstract error checking
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        size_t error_size;
        char* error_log = nullptr;
        nv_res = nvPTXCompilerGetErrorLogSize(compiler, &error_size);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get error log size, error code: {:d}", nv_res);
            return;
        }
        if (error_size != 0) {
            error_log = (char*)malloc(error_size + 1);
            assert(error_log != nullptr);
            nv_res = nvPTXCompilerGetErrorLog(compiler, error_log);
            if (nv_res != NVPTXCOMPILE_SUCCESS) {
                FATAL_ERROR("Could not get error log, error code: {:d}", nv_res);
                free(error_log);
                return;
            }
        }
        FATAL_ERROR("Could not compile PTX: \n{:s}", error_log);
        free(error_log);
        return;
    }

    nv_res = nvPTXCompilerGetCompiledProgramSize(compiler, &elf_size);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get compiled PTX elf size, error code: {:d}", nv_res);
        return;
    }

    elf = (char*)malloc(elf_size);
    assert(elf != nullptr);
    nv_res = nvPTXCompilerGetCompiledProgram(compiler, (void*)elf);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get compiled PTX elf data, error code: {:d}", nv_res);
        return;
    }

    // Dump Logs
    size_t info_size;
    nv_res = nvPTXCompilerGetInfoLogSize(compiler, &info_size);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get info log size, error code: {:d}", nv_res);
        return;
    }

    if (info_size != 0) {
        char* info_Log = (char*)malloc(info_size + 1);
        nv_res = nvPTXCompilerGetInfoLog(compiler, info_Log);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get PTX compiler logs, error code: {:d}", nv_res);
            free(info_Log);
            return;
        }
        INFO("PTX Compiler logs: \n{:s}", info_Log);
        free(info_Log);
    }

    // Cleanup compiler
    nv_res = nvPTXCompilerDestroy(&compiler);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not destroy compiler, error code: {:d}", nv_res);
        return;
    }

    // Extract kernels
    cu_res = cuModuleLoadDataEx(&module, elf, 0, nullptr, nullptr);
    if (cu_res != CUDA_SUCCESS) {
        FATAL_ERROR("Could not load module data from elf");
        return;
    }

    for (auto& kernel_name : kernel_names) {
        runtime_kernels.emplace(kernel_name, nullptr);
        cu_res = cuModuleGetFunction(&runtime_kernels[kernel_name], module, kernel_name.c_str());
        if (cu_res != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(cu_res, &errStr);
            FATAL_ERROR("ERROR IN cuModuleGetFunction for correlate: {:s}", errStr);
        }
    }

    free(elf);
}

void cudaCommand::record_start_event(int gpu_frame_id) {
    if (profiling) {
        CHECK_CUDA_ERROR(cudaEventCreate(&start_events[gpu_frame_id]));
        CHECK_CUDA_ERROR(
            cudaEventRecord(start_events[gpu_frame_id], device.getStream(cuda_stream_id)));
    }
}

cudaEvent_t cudaCommand::record_end_event(int gpu_frame_id) {
    CHECK_CUDA_ERROR(cudaEventCreate(&end_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(end_events[gpu_frame_id], device.getStream(cuda_stream_id)));
    return end_events[gpu_frame_id];
}
