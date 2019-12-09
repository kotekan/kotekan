#include "cudaCorrelatorRomein.hpp"
#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCorrelatorRomein);

cudaCorrelatorRomein::cudaCorrelatorRomein(Config& config, const string& unique_name,
                                           bufferContainer& host_buffers, cudaDeviceInterface& device) :
        cudaCommand(config, unique_name, host_buffers, device, "cudaCorrelator", "cudaCorrelator.cu") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    command_type = gpuCommandType::KERNEL;
    build();
}

cudaCorrelatorRomein::~cudaCorrelatorRomein() {}

cudaEvent_t cudaCorrelatorRomein::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    void *input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);

    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                          * _num_data_sets * sizeof(int32_t);
    void *output_memory = device.get_gpu_memory_array("output", gpu_frame_id, output_len);

    if (pre_event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_COMPUTE_STREAM), pre_event, 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(pre_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    INFO("Starting...");
    CUresult err;
    void *parameters[] = { &output_memory, &input_memory };
    int nblks_sq = (_num_elements / 128) * (_num_elements / 128 - 1) / 2;
    if (nblks_sq > 0) {
        err = cuLaunchKernel(sq_kernel,
                             2, nblks_sq, _num_local_freq,
                             32, 2, 2,
                             0,
                             device.getStream(CUDA_COMPUTE_STREAM),
                             parameters, NULL);
        if (err != CUDA_SUCCESS) {
            const char *errStr;
            cuGetErrorString(err, &errStr);
            INFO("ERROR IN cuLaunchKernel: {}", errStr);
        }
    }
    int nblks_tr = _num_elements/128;
    err = cuLaunchKernel(tr_kernel,
                         nblks_tr,_num_local_freq, 1,
                         32, 4, 1,
                         0,
                         device.getStream(CUDA_COMPUTE_STREAM),
                         parameters, NULL);
    if (err != CUDA_SUCCESS){
        const char *errStr;
        cuGetErrorString(err, &errStr);
        INFO("ERROR IN cuLaunchKernel: {}", errStr);
    }

    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(post_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    return post_events[gpu_frame_id];
}


// Specialist functions:
void cudaCorrelatorRomein::build() {
    std::string kernel_command="interleaveCorrelator";
    std::string kernel_file_name="../../lib/cuda/kernels/InterleaveCorrelator.cu";

    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvrtcResult res;

    DEBUG2("Building! {:s}", kernel_command)
    fp = fopen(kernel_file_name.c_str(), "r");
    if (fp == NULL) {
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

    res = nvrtcCreateProgram(&prog,program_buffer,kernel_command.c_str(),0,NULL,NULL);
    if (res != NVRTC_SUCCESS){
        const char*error_str = nvrtcGetErrorString (res);
        INFO("ERROR IN nvrtcCreateProgram: {}",error_str);
    }

    free(program_buffer);
    DEBUG2("Built! {:s}", kernel_command)

    std::string stations = fmt::format("-DNR_STATIONS={:d}",_num_elements/2);
    std::string channels = fmt::format("-DNR_CHANNELS={:d}",_num_local_freq);
    std::string nsamples = fmt::format("-DNR_SAMPLES_PER_CHANNEL={:d}",_samples_per_data_set);
    const char *opts[] = {"-arch=compute_75",
                          "-std=c++14",
                          //"-code=sm_75",
                          "-lineinfo",
                          //"-src-in-ptx",
                          "-DNR_BITS=4",
                          stations.c_str(),
                          channels.c_str(),
                          nsamples.c_str(),
                          "-DNR_POLARIZATIONS=2",
                          "-I/usr/local/cuda/include"
                          };
    res = nvrtcCompileProgram(prog,9, opts);
    if (res != NVRTC_SUCCESS){
        const char*error_str = nvrtcGetErrorString (res);
        ERROR("ERROR IN nvrtcCompileProgram: {}",error_str);
        // Obtain compilation log from the program.
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char *log = new char[logSize];
        nvrtcGetProgramLog(prog, log);
        INFO("COMPILE LOG: {}",log);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
//    char *ptx = new char[ptxSize];
    ptx = (char*)malloc(ptxSize);
    res = nvrtcGetPTX(prog, ptx);
    if (res != NVRTC_SUCCESS) {
        const char *error_str = nvrtcGetErrorString(res);
        INFO("ERROR IN nvrtcGetPTX: {}", error_str);
    }
    DEBUG2("PTX EXTRACTED");
    res = nvrtcDestroyProgram(&prog);
    if (res != NVRTC_SUCCESS) {
        const char *error_str = nvrtcGetErrorString(res);
        INFO("ERROR IN nvrtcDestroyProgram: {}", error_str);
    }

    CUresult err;
    // Get the kernel itself!
    err = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    if (err != CUDA_SUCCESS){
        const char *errStr;
        cuGetErrorString(err, &errStr);
        INFO("ERROR IN cuModuleLoadDataEx: {}", errStr);
    }
    err = cuModuleGetFunction(&sq_kernel, module, "correlateSquare");
    if (err != CUDA_SUCCESS){
        const char *errStr;
        cuGetErrorString(err, &errStr);
        INFO("ERROR IN cuModuleGetFunction for correlateSquare: {}",errStr);
    }
    err = cuModuleGetFunction(&tr_kernel, module, "correlateTriangle");
    if (err != CUDA_SUCCESS){
        const char *errStr;
        cuGetErrorString(err, &errStr);
        INFO("ERROR IN cuModuleGetFunction for correlateTriangle: {}", errStr);
    }

    DEBUG2("BUILT CUDA KERNEL! Yay!");
}
