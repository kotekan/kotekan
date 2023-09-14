#include "cudaQuantize.hpp"

#include "cudaUtils.hpp"
#include "math.h"
#include "mma.h"

void launch_quantize_kernel(cudaStream_t stream, int nframes, const __half2* in_base,
                            __half2* outf_base, unsigned int* outi_base, const int* index_array);

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaQuantize);

cudaQuantize::cudaQuantize(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device) {
    _num_chunks = config.get<int>(unique_name, "num_chunks");
    _gpu_mem_input = config.get<std::string>(unique_name, "gpu_mem_input");
    _gpu_mem_output = config.get<std::string>(unique_name, "gpu_mem_output");
    _gpu_mem_meanstd = config.get<std::string>(unique_name, "gpu_mem_meanstd");
    if (_num_chunks % FRAME_SIZE)
        throw std::runtime_error("The num_chunks parameter must be a factor of 32");
    std::string _gpu_mem_index = unique_name + "/index";

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaQuantize");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_input, true, true, false));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_meanstd, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(get_name() + "_index", false, true, true));

    size_t index_array_len = (size_t)_num_chunks * 2 * sizeof(int32_t);
    int32_t* index_array_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_index, index_array_len);

    int32_t* index_array_host = (int32_t*)malloc(index_array_len);
    assert(index_array_host);

    memset(index_array_host, 0, index_array_len);
    for (int f = 0; f < _num_chunks / FRAME_SIZE; f++) {
        for (int i = 0; i < FRAME_SIZE; i++) {
            // offset of the start of the chunk in the input array, for each chunk.
            index_array_host[f * FRAME_SIZE * 2 + i] = (f * FRAME_SIZE + i);
        }
        // offset for the mean/scale outputs per chunk
        index_array_host[f * FRAME_SIZE * 2 + FRAME_SIZE] = f * FRAME_SIZE;
        // offset for the output integers per frame;
        // this is in units of int32s, and the outputs are int4s, hence the divide by 8.
        index_array_host[f * FRAME_SIZE * 2 + FRAME_SIZE + 1] = f * FRAME_SIZE * (CHUNK_SIZE / 8);
    }

    CHECK_CUDA_ERROR(
        cudaMemcpy(index_array_memory, index_array_host, index_array_len, cudaMemcpyHostToDevice));
    free(index_array_host);
}

cudaQuantize::~cudaQuantize() {}

cudaEvent_t cudaQuantize::execute(cudaPipelineState& pipestate,
                                  const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);

    size_t input_frame_len = (size_t)_num_chunks * CHUNK_SIZE * sizeof(float16_t);
    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_input, pipestate.gpu_frame_id, input_frame_len);
    INFO("Input frame length: {:d} x {:d} x 2 = {:d}", _num_chunks, CHUNK_SIZE, input_frame_len);
    //  divide by 2 because of packed int4 outputs
    size_t output_frame_len = (size_t)_num_chunks * CHUNK_SIZE / 2;
    INFO("Output frame length: {:d} x {:d} / 2 = {:d}", _num_chunks, CHUNK_SIZE, output_frame_len);
    int32_t* output_memory = (int32_t*)device.get_gpu_memory_array(
        _gpu_mem_output, pipestate.gpu_frame_id, output_frame_len);

    size_t meanstd_frame_len = (size_t)_num_chunks * 2 * sizeof(float16_t);
    void* meanstd_memory =
        device.get_gpu_memory_array(_gpu_mem_meanstd, pipestate.gpu_frame_id, meanstd_frame_len);

    std::string _gpu_mem_index = unique_name + "/index";
    size_t index_array_len = (size_t)_num_chunks * 2 * sizeof(int32_t);
    int32_t* index_array_memory = (int32_t*)device.get_gpu_memory(_gpu_mem_index, index_array_len);

    record_start_event(pipestate.gpu_frame_id);

    launch_quantize_kernel(device.getStream(cuda_stream_id), _num_chunks / FRAME_SIZE,
                           (const __half2*)input_memory, (__half2*)meanstd_memory,
                           (unsigned int*)output_memory, index_array_memory);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return record_end_event(pipestate.gpu_frame_id);
}
