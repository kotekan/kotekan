#include "cudaBasebandBeamformer.hpp"

#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

#define CHECK_CU_ERROR(result)                                                                     \
    if (result != CUDA_SUCCESS) {                                                                  \
        const char* errstr = NULL;                                                                 \
        cuGetErrorString(result, &errstr);                                                         \
        internal_logging(LOG_ERR, __log_prefix, "Error at {:s}:{:d}; Error type: {:s}", __FILE__,  \
                         __LINE__, errstr);                                                        \
        std::abort();                                                                              \
    }


REGISTER_CUDA_COMMAND(cudaBasebandBeamformer);

cudaBasebandBeamformer::cudaBasebandBeamformer(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "baseband_beamformer", "bb1.ptx") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_beams = config.get<int>(unique_name, "num_beams");
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_output_scaling = config.get<std::string>(unique_name, "gpu_mem_output_scaling");
    _gpu_mem_formed_beams = config.get<std::string>(unique_name, "gpu_mem_formed_beams");

    command_type = gpuCommandType::KERNEL;

    std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_name}, opts);

    /*
    std::vector<std::string> opts = {
        "-arch=compute_86",
        "-std=c++17",
        "-lineinfo",
        "-DNR_BITS=4",
        fmt::format("-DNR_RECEIVERS={:d}", _num_elements / 2),
        fmt::format("-DNR_CHANNELS={:d}", _num_local_freq),
        fmt::format("-DNR_SAMPLES_PER_CHANNEL={:d}", _samples_per_data_set),
        fmt::format("-DNR_RECEIVERS_PER_BLOCK={:d}", _elements_per_thread_block / 2),
        "-DNR_POLARIZATIONS=2",
        "-I/usr/local/cuda/include"};
    build({"correlate"}, opts);
    */

    // HACK

    size_t phase_len = (size_t)_num_elements * _num_local_freq * _num_beams * 2;
    int8_t* phase_memory = (int8_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);

    int8_t* cpu_phase_memory = (int8_t*)malloc(phase_len);
    for (size_t i = 0; i < phase_len; i++)
        cpu_phase_memory[i] = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(phase_memory, cpu_phase_memory, phase_len, cudaMemcpyHostToDevice));
    free(cpu_phase_memory);

    size_t shift_len = (size_t)_num_local_freq * _num_beams * 2 * sizeof(int32_t);
    int32_t* shift_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_output_scaling, shift_len);

    int32_t* cpu_shift_memory = (int32_t*)malloc(shift_len);
    for (size_t i = 0; i < shift_len / sizeof(int32_t); i++)
        cpu_shift_memory[i] = 1;
    CHECK_CUDA_ERROR(cudaMemcpy(shift_memory, cpu_shift_memory, shift_len, cudaMemcpyHostToDevice));
    free(cpu_shift_memory);
}

cudaBasebandBeamformer::~cudaBasebandBeamformer() {}

cudaEvent_t cudaBasebandBeamformer::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    size_t voltage_len = (size_t)_num_elements * _num_local_freq * _samples_per_data_set;
    void* voltage_memory = device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, voltage_len);

    size_t phase_len = (size_t)_num_elements * _num_local_freq * _num_beams * 2;
    // int8_t* phase_memory = (int8_t*)device.get_gpu_memory_array(_gpu_mem_phase, gpu_frame_id,
    // phase_len);
    int8_t* phase_memory = (int8_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);

    size_t shift_len = (size_t)_num_local_freq * _num_beams * 2 * sizeof(int32_t);
    // int32_t* shift_memory = (int32_t*)device.get_gpu_memory_array(_gpu_mem_output_scaling,
    // gpu_frame_id, shift_len);
    int32_t* shift_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_output_scaling, shift_len);

    size_t output_len = (size_t)_num_local_freq * _num_beams * _samples_per_data_set * 2;
    void* output_memory =
        device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id, output_len);

    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_COMPUTE_STREAM), pre_event, 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(
        cudaEventRecord(pre_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    CUresult err;
    // void* parameters[] = {&voltage_memory, &phase_memory, &shift_memory, &output_memory};
    // A, E, s, J
    void* parameters[] = {&phase_memory, &voltage_memory, &shift_memory, &output_memory};

    // threads = (32,4)
    // blocks per sm = 8
    // nblocks = 84 * 8

    int shared_mem_bytes = 13568 * sizeof(int32_t);

    INFO("Kernel: {:p}", (void*)runtime_kernels[kernel_name]);

    /*
      cudaFuncAttributes attrs;
      CHECK_CUDA_ERROR(cudaFuncGetAttributes(&attrs, &runtime_kernels[kernel_name]));
    */
    int attr = 0;
    CHECK_CU_ERROR(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                      runtime_kernels[kernel_name]));
    INFO("Max threads per block: {}", attr);

    /*
      CHECK_CUDA_ERROR(cudaFuncSetAttribute(runtime_kernels[kernel_name],
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_mem_bytes));
    */
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shared_mem_bytes));

    attr = 0;
    CHECK_CU_ERROR(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      runtime_kernels[kernel_name]));
    INFO("Max dynamic shared memory size: {}", attr);

    err = cuLaunchKernel(runtime_kernels[kernel_name], 84 * 8, 1, 1, 32, 4, 1, shared_mem_bytes,
                         device.getStream(CUDA_COMPUTE_STREAM), parameters, NULL);
    /*
      CUlaunchConfig config;
      config.blockDim.x = 84*8;
      config.blockDim.y = 1;
      config.blockDim.z = 1;
      config.gridDim.x = 32;
      config.gridDim.y = 4;
      config.gridDim.z = 1;
      config.dynamicSmemBytes = shared_mem_bytes;
      config.stream = device.getStream(CUDA_COMPUTE_STREAM);
      err = cuLaunchKernelEx(&config, &runtime_kernels[kernel_name], parameters, NULL);
    */

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("Error number: {}", err);
        ERROR("ERROR IN cuLaunchKernel: {}", errStr);
    }

    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(
        cudaEventRecord(post_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    return post_events[gpu_frame_id];
}
