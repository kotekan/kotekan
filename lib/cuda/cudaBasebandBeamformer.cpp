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
    cudaCommand(config, unique_name, host_buffers, device, "baseband_beamformer", "bb.ptx") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_beams = config.get<int>(unique_name, "num_beams");
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_phase = config.get<std::string>(unique_name, "gpu_mem_phase");
    _gpu_mem_output_scaling = config.get<std::string>(unique_name, "gpu_mem_output_scaling");
    _gpu_mem_formed_beams = config.get<std::string>(unique_name, "gpu_mem_formed_beams");
    _gpu_mem_info = config.get<std::string>(unique_name, "gpu_mem_info");

    command_type = gpuCommandType::KERNEL;

    std::vector<std::string> opts = {
        "--gpu-name=sm_86",
        "--verbose",
    };
    build_ptx({kernel_name}, opts);

    // HACK -- this was for setting constant phase & shift arrays in local arrays
    // (not managed by kotekan)
    /*
    size_t phase_len = (size_t)_num_elements * _num_local_freq * _num_beams * 2;
    int8_t* phase_memory = (int8_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);

    int8_t* cpu_phase_memory = (int8_t*)malloc(phase_len);
    for (size_t i = 0; i < phase_len; i++)
        cpu_phase_memory[i] = 0x1;

    CHECK_CUDA_ERROR(cudaMemcpy(phase_memory, cpu_phase_memory, phase_len, cudaMemcpyHostToDevice));
    free(cpu_phase_memory);

    size_t shift_len = (size_t)_num_local_freq * _num_beams * 2 * sizeof(int32_t);
    int32_t* shift_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_output_scaling, shift_len);

    int32_t* cpu_shift_memory = (int32_t*)malloc(shift_len);
    for (size_t i = 0; i < shift_len / sizeof(int32_t); i++)
        cpu_shift_memory[i] = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(shift_memory, cpu_shift_memory, shift_len, cudaMemcpyHostToDevice));
    free(cpu_shift_memory);
    */
}

cudaBasebandBeamformer::~cudaBasebandBeamformer() {}


// This struct is Erik's interpretation of what Julia is expecting for its "CuDevArray" type.
template<typename T, int64_t N>
struct CuDeviceArray {
    T* ptr;
    int64_t maxsize;
    int64_t dims[N];
    int64_t len;
};
typedef CuDeviceArray<int32_t,1> kernel_arg;

cudaEvent_t cudaBasebandBeamformer::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    size_t voltage_len = (size_t)_num_elements * _num_local_freq * _samples_per_data_set;
    void* voltage_memory = device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, voltage_len);

    size_t phase_len = (size_t)_num_elements * _num_local_freq * _num_beams * 2;
    // TEMPORARY -- get our local phase memory, not from kotekan
    //int8_t* phase_memory = (int8_t*)device.get_gpu_memory(_gpu_mem_phase, phase_len);
    int8_t* phase_memory = (int8_t*)device.get_gpu_memory_array(_gpu_mem_phase, gpu_frame_id,
                                                                phase_len);

    size_t shift_len = (size_t)_num_local_freq * _num_beams * 2 * sizeof(int32_t);
    // TEMPORARY -- get our local shift memory, not from kotekan
    //int32_t* shift_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_output_scaling, shift_len);
    int32_t* shift_memory = (int32_t*)device.get_gpu_memory_array(_gpu_mem_output_scaling,
                                                                  gpu_frame_id, shift_len);

    size_t output_len = (size_t)_num_local_freq * _num_beams * _samples_per_data_set * 2;

    void* output_memory =
        device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id, output_len);

    size_t info_len = (size_t)(threads_x * threads_y * blocks_x * sizeof(int32_t));
    //int32_t* info_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_info, info_len);
    int32_t* info_memory = (int32_t*)device.get_gpu_memory_array(_gpu_mem_info,
                                                                 gpu_frame_id, info_len);

    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_COMPUTE_STREAM), pre_event, 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(
        cudaEventRecord(pre_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    CUresult err;
    // A, E, s, J
    const char* exc = "exception";
    kernel_arg arr[5];

    arr[0].ptr = (int32_t*)phase_memory;
    arr[0].maxsize = phase_len;
    arr[0].dims[0] = phase_len;
    arr[0].len = phase_len;

    arr[1].ptr = (int32_t*)voltage_memory;
    arr[1].maxsize = voltage_len;
    arr[1].dims[0] = voltage_len;
    arr[1].len = voltage_len;

    arr[2].ptr = shift_memory;
    arr[2].maxsize = shift_len;
    arr[2].dims[0] = shift_len / sizeof(int32_t);
    arr[2].len = shift_len / sizeof(int32_t);

    arr[3].ptr = (int32_t*)output_memory;
    arr[3].maxsize = output_len;
    arr[3].dims[0] = output_len;
    arr[3].len = output_len;

    arr[4].ptr = (int32_t*)info_memory;
    arr[4].maxsize = info_len;
    arr[4].dims[0] = info_len / sizeof(int32_t);
    arr[4].len = info_len / sizeof(int32_t);

    void* parameters[] = {
        &(exc),
        &(arr[0]),
        &(arr[1]),
        &(arr[2]),
        &(arr[3]),
        &(arr[4]),
    };

    DEBUG("Kernel_name: {}", kernel_name);
    DEBUG("runtime_kernels[kernel_name]: {}", (void*)runtime_kernels[kernel_name]);
    CHECK_CU_ERROR(cuFuncSetAttribute(runtime_kernels[kernel_name],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      shared_mem_bytes));

    DEBUG("Running CUDA Baseband Beamformer on GPU frame {:d}", gpu_frame_id);
    err = cuLaunchKernel(runtime_kernels[kernel_name],
                         blocks_x, blocks_y, 1,
                         threads_x, threads_y, 1, shared_mem_bytes,
                         device.getStream(CUDA_COMPUTE_STREAM), parameters, NULL);

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
