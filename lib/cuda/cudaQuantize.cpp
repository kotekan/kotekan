#include "cudaQuantize.hpp"

#include "cudaUtils.hpp"

#include "math.h"
#include "mma.h"

/*
 void launch_quantize_kernel(cudaStream_t stream, int nframes, const __half2 *in_base,
                            __half2 *outf_base, unsigned int *outi_base,
                            const int *index_array);
 */
void launch_quantize_kernel(cudaStream_t stream, int nframes, const void *in_base,
                            void *outf_base, int32_t *outi_base,
                            const int32_t *index_array);

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaQuantize);

cudaQuantize::cudaQuantize(Config& config, const std::string& unique_name,
                               bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaQuantize", "") {
    _num_chunks = config.get<int>(unique_name, "num_chunks");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _gpu_mem_meanstd = config.get<std::string>(unique_name, "gpu_mem_meanstd");
    if (_num_chunks % FRAME_SIZE)
        throw std::runtime_error("The num_chunks parameter must be a factor of 32");
    set_command_type(gpuCommandType::KERNEL);

    size_t index_array_len = _num_chunks * 2 * sizeof(int32_t);
    int32_t* index_array_memory =
        device.get_gpu_memory(_gpu_mem_input, index_array_len);

    int32_t* index_array_host = (int32_t*)malloc(index_array_len);
    assert(index_array_host);

    memset(index_array_host, 0, index_array_len);
    for (int f = 0; f < _num_chunks/FRAME_SIZE; f++) {
        for (int i = 0; i < FRAME_SIZE; i++) {
            // offset of the start of the chunk in the input array, for each chunk.
            index_array_host[f*FRAME_SIZE*2 + i] = (f * FRAME_SIZE + i) * CHUNK_SIZE;
        }
        // offset for the mean/scale outputs per chunk
        index_array_host[f*FRAME_SIZE*2 + FRAME_SIZE] = f * FRAME_SIZE * 2;
        // offset for the output integers per frame;
        // this is in units of int32s, and the outputs are int4s, hence the divide by 8.
        index_array_host[f*FRAME_SIZE*2 + FRAME_SIZE + 1] = f * FRAME_SIZE * (CHUNK_SIZE / 8);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(index_array_memory, index_array_host, index_array_len,
                                cudaMemcpyHostToDevice));
    free(index_array_host);
}

cudaQuantize::~cudaQuantize() {}

cudaEvent_t cudaQuantize::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events) {
    pre_execute(gpu_frame_id);

    size_t input_frame_len = _num_chunks * CHUNK_SIZE * sizeof(float16_t);
    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_input, gpu_frame_id, input_frame_len);

    //  divide by 2 because of packed int4 outputs
    size_t output_frame_len = _num_chunks * CHUNK_SIZE / 2;
    void* output_memory =
        device.get_gpu_memory_array(_gpu_mem_output, gpu_frame_id, output_frame_len);

    size_t meanstd_frame_len = _num_chunks * 2 * sizeof(float16_t);
    void* meanstd_memory =
        device.get_gpu_memory_array(_gpu_mem_meanstd, gpu_frame_id, meanstd_frame_len);

    size_t index_array_len = _num_chunks * 2 * sizeof(int32_t);
    int32_t* index_array_memory =
        device.get_gpu_memory(_gpu_mem_input, index_array_len);

    record_start_event(gpu_frame_id);

    launch_quantize_kernel(device.getStream(cuda_stream_id),
                           _num_chunks / FRAME_SIZE,
                           input_memory, meanstd_memory, output_memory,
                           index_array_memory);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event(gpu_frame_id);
}
