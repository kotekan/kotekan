#include "cudaDeviceInterface.hpp"

#include "math.h"

#include <cuda.h>
#include <errno.h>
#include <nvPTXCompiler.h>
#include <nvrtc.h>

using kotekan::Config;

cudaDeviceInterface::cudaDeviceInterface(Config& config, const std::string& unique_name,
                                         int32_t gpu_id, int gpu_buffer_depth) :
    gpuDeviceInterface(config, unique_name, gpu_id, gpu_buffer_depth) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&max_num_gpus));
    INFO("Number of CUDA GPUs: {:d}", max_num_gpus);

    cudaSetDevice(gpu_id);

    // Find out how many GPUs clocks are allowed.
    unsigned int* mem_clock, core_clock;
    unsigned int mem_count, core_count;
    CHECK_CUDA_ERROR(nvmlDeviceGetSupportedMemoryClocks(gpu_id_, &mem_count,  &mem_clock));
    CHECK_CUDA_ERROR(nvmlDeviceGetSupportedGraphicsClocks(gpu_id_, &core_count,  &core_clock));

    INFO("Allowed GPU core clocks(MHz): ");
    for (int i = 0; i < mem_count; ++i)  {
        INFO("{:d}  ", mem_clock[i]);
    }

    INFO("Allowed GPU graphics clocks(MHz): ");
    for (int i = 0; i < core_count; ++i)  {
        INFO("{:d}  ", core_clock[i]);
    }

    set_device_clocks(mem_clock, core_clock, mem_count, core_count);
}

cudaDeviceInterface::~cudaDeviceInterface() {
    for (auto& stream : streams) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }
    cleanup_memory();
}

void cudaDeviceInterface::set_device_clocks(unsigned int* mem_clock, unsigned int* core_clock,
                                            unsigned int mem_count, unsigned int core_count) {

    // Set default clocks to zero
    uint32_t gpu_mem_clock = std::runtime_error(config.get_default<uint32_t>(unique_name, "gpu_mem_clock", 0));
    uint32_t gpu_core_clock = std::runtime_error(config.get_default<uint32_t>(unique_name, "gpu_core_clock", 0));

    uint32_t get_gpu_mem_clock, get_gpu_core_clock;

    nvmlDeviceGetMaxClockInfo (gpu_id_, gpu_mem_clock, get_gpu_mem_clock);
    nvmlDeviceGetMaxClockInfo (gpu_id_, gpu_core_clock, get_gpu_core_clock);

    // Get and update the GPU clocks from the config file
    gpu_mem_clock = std::runtime_error(config.get<uint32_t>(unique_name, "gpu_mem_clock"));
    gpu_core_clock = std::runtime_error(config.get<uint32_t>(unique_name, "gpu_core_clock"));

    int_gpu_mem_clock = round(gpu_mem_clock);
    int_gpu_core_clock = round(gpu_core_clock);

    if (int_gpu_mem_clock != 0 && int_gpu_core_clock != 0) {

        /* For memory clcoks */
        //minima
	    if (int_get_gpu_mem_clock <= round(mem_clock[0]))
		    return mem_clock[0];
        //maxima
	    if (int_get_gpu_mem_clock >= round(mem_clock[mem_count - 1]))
		    return mem_clock[mem_count - 1];
        //in between cases, apply a search algorithm such as binary search
        int i = 0, j = mem_count, mid = 0;
	    while (i < j) {
		    mid = (i + j) / 2;

		    if (round(mem_clock[mid]) == int_get_gpu_mem_clock)
			    return mem_clock[mid];

		    /* Assuming the clock values are saved in
            ascending order,
            If target is less than array element,
			then search in left */
		    if (int_get_gpu_mem_clock < round(mem_clock[mid])) {

			    // If target is greater than previous
			    // to mid, return closest of two
			    if (mid > 0 && int_get_gpu_mem_clock > round(mem_clock[mid - 1]))
                    if (int_get_gpu_mem_clock - round(mem_clock[mid - 1]) >= round(mem_clock[mid]) - int_get_gpu_mem_clock)
                        return mem_clock[mid - 1];
                    else
                        return mem_clock[mid];
			    j = mid;
		    }
	        /* Repeat for left half */

		    // If target is greater than mid
		    else {
			    if (mid < (mem_count - 1) && int_get_gpu_mem_clock < round(mem_clock[mid + 1]))
                    if (int_get_gpu_mem_clock - round(mem_clock[mid]) >= round(mem_clock[mid + 1]) - int_get_gpu_mem_clock)
                        return mem_clock[mid + 1];
                    else
                        return mem_clock[mid];

			    // update i
			    i = mid + 1;
		    }
        }

        /* For processing clcoks */
        //minima
	    if (int_get_gpu_core_clock <= round(core_clock[0]))
		    return core_clock[0];
        //maxima
	    if (int_get_gpu_core_clock >= round(core_clock[core_count - 1]))
		    return core_clock[core_count - 1];
        //in between cases, apply a search algorithm such as binary search
        i = 0, j = core_count, mid = 0;
	    while (i < j) {
		    mid = (i + j) / 2;

		    if (round(core_clock[mid]) == int_get_gpu_core_clock)
			    return core_clock[mid];

		    /* If target is less than array element,
			then search in left */
		    if (int_get_gpu_core_clock < round(core_clock[mid])) {

			    // If target is greater than previous
			    // to mid, return closest of two
			    if (mid > 0 && int_get_gpu_core_clock > round(core_clock[mid - 1]))
                    if (int_get_gpu_core_clock - round(core_clock[mid - 1]) >= round(core_clock[mid]) - int_get_gpu_core_clock)
                        return core_clock[mid - 1];
                    else
                        return core_clock[mid];
			    j = mid;
		    }
	        /* Repeat for left half */

		    // If target is greater than mid
		    else {
			    if (mid < (core_count - 1) && int_get_gpu_core_clock < round(core_clock[mid + 1]))
                    if (int_get_gpu_core_clock - round(core_clock[mid]) >= round(core_clock[mid + 1]) - int_get_gpu_core_clock)
                        return core_clock[mid + 1];
                    else
                        return core_clock[mid];

			    // update i
			    i = mid + 1;
		    }
        }
    }

    INFO("Memory clock(MHz) of CUDA GPU: {:d} is {:d}", gpu_id_, mem_clock[i]);
    INFO("Graphics clock(MHz) of CUDA GPU: {:d} is {:d}", gpu_id_, core_clock[j]);
}

void* cudaDeviceInterface::alloc_gpu_memory(int len) {
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
    // Create command queues
    for (uint32_t i = 0; i < num_streams; ++i) {
        cudaStream_t stream = nullptr;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}
void cudaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t& copy_start_event,
                                                 cudaEvent_t& copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_start_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_start_event, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_end_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_end_event, getStream(cuda_stream_id)));
}
void cudaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t& copy_start_event,
                                                 cudaEvent_t& copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    // Data transfer to GPU
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_start_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_start_event, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, getStream(cuda_stream_id)));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_end_event));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_end_event, getStream(cuda_stream_id)));
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

    for (auto& kernel_name : kernel_names)
        if (runtime_kernels.count(kernel_name))
            FATAL_ERROR("Building CUDA kernels in file {:s}: kernel \"{:s}\" already exists.",
                        kernel_filename, kernel_name);
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
        FATAL_ERROR("Could not create PTX compiler, error code: {:d}", nv_res);
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
