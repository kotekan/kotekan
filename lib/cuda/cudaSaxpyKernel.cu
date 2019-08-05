#include "cudaSaxpyKernel.cuh"
#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSaxpyKernel);

cudaSaxpyKernel::cudaSaxpyKernel(Config& config, const string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaSaxpy", "cudaSaxpy.cu") {
    _saxpy_scale = config.get_default<float>(unique_name,"saxpy_scale", 1.0f);
    _data_length = config.get<int>(unique_name,"data_length");

    command_type = gpuCommandType::KERNEL;
}

cudaSaxpyKernel::~cudaSaxpyKernel() {}

__global__
void saxpy(float a, float *x, float *y)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
   y[threadId] = a*x[threadId] + y[threadId];
}

cudaEvent_t cudaSaxpyKernel::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    void *input_memory = device.get_gpu_memory_array("input", gpu_frame_id, _data_length);
    void *output_memory = device.get_gpu_memory_array("output", gpu_frame_id, _data_length);

    if (pre_event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_COMPUTE_STREAM), pre_event, 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(pre_events[gpu_frame_id]));

    dim3 tbp (32,1,1);
    dim3 blk (_data_length/32/sizeof(float),1,1);
    saxpy<<<blk,tbp,0,device.getStream(CUDA_COMPUTE_STREAM)>>>(_saxpy_scale, (float*)input_memory, (float*)output_memory);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(post_events[gpu_frame_id]));

    return post_events[gpu_frame_id];
}
