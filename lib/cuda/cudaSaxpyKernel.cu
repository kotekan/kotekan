#include "cudaSaxpyKernel.cuh"
#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaSaxpyKernel);

cudaSaxpyKernel::cudaSaxpyKernel(Config& config, const string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaSaxpy", "cudaSaxpy.cu") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    command_type = gpuCommandType::KERNEL;
}

cudaSaxpyKernel::~cudaSaxpyKernel() {}

__global__
void saxpy(int a, int *x, int *y)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                             + blockIdx.z * gridDim.x * gridDim.y;
    int threadId = blockId * (blockDim.x  * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * (blockDim.x))
            + threadIdx.x;
   y[threadId] = a*x[threadId];// + y[i];
}

cudaEvent_t cudaSaxpyKernel::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    void *input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);

    uint32_t output_len = _num_elements * _num_local_freq * _samples_per_data_set;
    void *output_memory = device.get_gpu_memory_array("output", gpu_frame_id, output_len);

    if (pre_event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_COMPUTE_STREAM), pre_event, 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(pre_events[gpu_frame_id]));

    dim3 tbp (16,16,4);
    dim3 blk (_num_elements/16, _num_local_freq/16, _samples_per_data_set/16);
    saxpy<<<blk,tbp,0,device.getStream(CUDA_COMPUTE_STREAM)>>>(1.0f, (int*)input_memory, (int*)output_memory);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(post_events[gpu_frame_id]));

    return post_events[gpu_frame_id];
}
