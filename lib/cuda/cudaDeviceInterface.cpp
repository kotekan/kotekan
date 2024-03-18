#include "cudaDeviceInterface.hpp"

#include "math.h"

#include <cuda.h>
#include <errno.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>

using kotekan::Config;

std::map<int, std::shared_ptr<cudaDeviceInterface>> cudaDeviceInterface::inst_map;

std::shared_ptr<cudaDeviceInterface>
cudaDeviceInterface::get(int32_t gpu_id, const std::string& name, Config& config) {
    if (inst_map.count(gpu_id) == 0)
        inst_map[gpu_id] = std::make_shared<cudaDeviceInterface>(config, name, gpu_id);
    return inst_map[gpu_id];
}

cudaDeviceInterface::cudaDeviceInterface(Config& config, const std::string& unique_name,
                                         int32_t gpu_id) :
    gpuDeviceInterface(config, unique_name, gpu_id) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&max_num_gpus));
    INFO("Number of CUDA GPUs: {:d}", max_num_gpus);

    if (gpu_id > max_num_gpus) {
        throw std::runtime_error(
            "Asked for a GPU ID which is higher than the maximum number of GPUs in the system");
    }

    set_thread_device();
}

cudaDeviceInterface::~cudaDeviceInterface() {
    for (auto& stream : streams) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }
    cleanup_memory();
}

void cudaDeviceInterface::set_thread_device() {
    CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
}

void* cudaDeviceInterface::alloc_gpu_memory(size_t len) {
    void* ret;
    CHECK_CUDA_ERROR(cudaMalloc(&ret, len));
    return ret;
}
void cudaDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

cudaStream_t cudaDeviceInterface::getStream(int32_t cuda_stream_id) {
    return streams[cuda_stream_id];
}

int32_t cudaDeviceInterface::get_num_streams() {
    return streams.size();
}

void cudaDeviceInterface::prepareStreams(uint32_t num_streams) {
    // Create GPU command queues
    for (uint32_t i = streams.size(); i < num_streams; ++i) {
        cudaStream_t stream = nullptr;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}
void cudaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t* copy_start_event,
                                                 cudaEvent_t* copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    if (copy_start_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_start_event, getStream(cuda_stream_id)));
    }
    // Data transfer to GPU
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, getStream(cuda_stream_id)));
    if (copy_end_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_end_event, getStream(cuda_stream_id)));
    }
}
void cudaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t* copy_start_event,
                                                 cudaEvent_t* copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    if (copy_start_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_start_event, getStream(cuda_stream_id)));
    }
    // Data transfer from GPU
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, getStream(cuda_stream_id)));
    if (copy_end_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_end_event, getStream(cuda_stream_id)));
    }
}

void cudaDeviceInterface::build(const std::string& kernel_filename,
                                const std::vector<std::string>& kernel_names,
                                const std::vector<std::string>& opts) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvrtcResult res;

    for (auto& kernel_name : kernel_names)
        if (runtime_kernels.count(kernel_name))
            FATAL_ERROR("Building CUDA kernels in file {:s}: kernel \"{:s}\" already exists.",
                        kernel_filename, kernel_name);

    // DEBUG("Building! {:s}", kernel_command)
    //  Load the kernel file contents into `program_buffer`
    fp = fopen(kernel_filename.c_str(), "r");
    if (fp == nullptr) {
        FATAL_ERROR("error loading file: {:s}", kernel_filename.c_str());
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);

    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
    if (sizeRead < (int32_t)program_size)
        FATAL_ERROR("Error reading the file: {:s}", kernel_filename);
    fclose(fp);

    // Create the program object
    nvrtcProgram prog;
    res = nvrtcCreateProgram(&prog, program_buffer, nullptr, 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        INFO("ERROR IN nvrtcCreateProgram: {}", error_str);
    }

    free(program_buffer);

    // Convert compiler options to a c-style array.
    std::vector<const char*> cstrings;
    cstrings.reserve(opts.size());

    for (auto& s : opts)
        cstrings.push_back(s.c_str());

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

void cudaDeviceInterface::build_ptx(const std::string& kernel_filename,
                                    const std::vector<std::string>& kernel_names,
                                    const std::vector<std::string>& opts,
                                    const std::string& kernel_name_prefix) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvPTXCompileResult nv_res;
    CUresult cu_res;
    nvPTXCompilerHandle compiler = nullptr;
    size_t elf_size;
    char* elf;
    CUmodule module;

    for (auto& kernel_name : kernel_names)
        if (runtime_kernels.count(kernel_name_prefix + kernel_name))
            FATAL_ERROR("Building CUDA kernels in file {:s}: kernel \"{:s}\" already exists.",
                        kernel_filename, kernel_name_prefix + kernel_name);
    // DEBUG("Building! {:s}", kernel_command)

    // Load the kernel file contents into `program_buffer`
    fp = fopen(kernel_filename.c_str(), "r");
    if (fp == NULL) {
        FATAL_ERROR("error loading file: {:s}", kernel_filename.c_str());
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);

    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
    if (sizeRead < (int32_t)program_size)
        FATAL_ERROR("Error reading the file: {:s}", kernel_filename);
    fclose(fp);

    // Create the compiler
    nv_res = nvPTXCompilerCreate(&compiler, program_size, program_buffer);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        // TODO Report ENUM names.
        FATAL_ERROR("Could not create PTX compiler, error code: {:d}", (int)nv_res);
        return;
    }

    // Convert compiler options to a c-style array.
    std::vector<const char*> cstring_opts;
    cstring_opts.reserve(opts.size());
    for (auto& s : opts)
        cstring_opts.push_back(s.c_str());

    // Compile the code
    nv_res = nvPTXCompilerCompile(compiler, cstring_opts.size(), cstring_opts.data());
    // TODO Abstract error checking
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        size_t error_size;
        char* error_log = nullptr;
        nv_res = nvPTXCompilerGetErrorLogSize(compiler, &error_size);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get error log size, error code: {:d}", (int)nv_res);
            return;
        }
        if (error_size != 0) {
            error_log = (char*)malloc(error_size + 1);
            assert(error_log != nullptr);
            nv_res = nvPTXCompilerGetErrorLog(compiler, error_log);
            if (nv_res != NVPTXCOMPILE_SUCCESS) {
                FATAL_ERROR("Could not get error log, error code: {:d}", (int)nv_res);
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
        FATAL_ERROR("Could not get compiled PTX elf size, error code: {:d}", (int)nv_res);
        return;
    }

    elf = (char*)malloc(elf_size);
    assert(elf != nullptr);
    nv_res = nvPTXCompilerGetCompiledProgram(compiler, (void*)elf);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get compiled PTX elf data, error code: {:d}", (int)nv_res);
        return;
    }

    // Dump Logs
    size_t info_size;
    nv_res = nvPTXCompilerGetInfoLogSize(compiler, &info_size);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get info log size, error code: {:d}", (int)nv_res);
        return;
    }

    if (info_size != 0) {
        char* info_Log = (char*)malloc(info_size + 1);
        nv_res = nvPTXCompilerGetInfoLog(compiler, info_Log);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get PTX compiler logs, error code: {:d}", (int)nv_res);
            free(info_Log);
            return;
        }
        INFO("PTX Compiler logs: \n{:s}", info_Log);
        free(info_Log);
    }

    // Cleanup compiler
    nv_res = nvPTXCompilerDestroy(&compiler);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not destroy compiler, error code: {:d}", (int)nv_res);
        return;
    }

    // Extract kernels
    cu_res = cuModuleLoadDataEx(&module, elf, 0, nullptr, nullptr);
    if (cu_res != CUDA_SUCCESS) {
        FATAL_ERROR("Could not load module data from elf");
        return;
    }

    for (auto& kernel_name : kernel_names) {
        runtime_kernels.emplace(kernel_name_prefix + kernel_name, nullptr);
        cu_res = cuModuleGetFunction(&runtime_kernels[kernel_name_prefix + kernel_name], module,
                                     kernel_name.c_str());
        if (cu_res != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(cu_res, &errStr);
            FATAL_ERROR("ERROR IN cuModuleGetFunction for correlate: {:s}", errStr);
        }
    }

    free(elf);
}
