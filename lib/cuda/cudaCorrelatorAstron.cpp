#include "cudaCorrelatorAstron.hpp"

#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCorrelatorAstron);

cudaCorrelatorAstron::cudaCorrelatorAstron(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers,
                                           cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "correlate", "TCCorrelator.cu") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _block_size = config.get_default<int>(unique_name, "block_size", 2);
    _elements_per_thread_block =
        config.get_default<int>(unique_name, "elements_per_thread_block", 128);
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_correlation_matrix =
        config.get<std::string>(unique_name, "gpu_mem_correlation_matrix");

    set_command_type(gpuCommandType::KERNEL);

    if (_block_size != 2)
        throw std::runtime_error("The block size must be 2 for the Astron TC kernels");

    if (_elements_per_thread_block != 64 && _elements_per_thread_block != 96
        && _elements_per_thread_block != 128)
        throw std::runtime_error("elements_per_thread_block must be one of 64, 96, 128");

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
}

cudaCorrelatorAstron::~cudaCorrelatorAstron() {}

cudaEvent_t cudaCorrelatorAstron::execute(int gpu_frame_id,
                                          const std::vector<cudaEvent_t>& pre_events) {
    pre_execute(gpu_frame_id);

    size_t input_frame_len = (size_t)_num_elements * _num_local_freq * _samples_per_data_set;
    void* input_memory = device.get_gpu_memory(_gpu_mem_voltage, input_frame_len);

    size_t output_len = (size_t)_num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                        * _num_data_sets * sizeof(int32_t);
    void* output_memory =
        device.get_gpu_memory_array(_gpu_mem_correlation_matrix, gpu_frame_id, output_len);

    record_start_event(gpu_frame_id);

    CUresult err;
    void* parameters[] = {&output_memory, &input_memory};

    int num_thread_blocks =
        (_num_elements / _elements_per_thread_block) * (_num_elements / _elements_per_thread_block);
    err = cuLaunchKernel(runtime_kernels["correlate"], num_thread_blocks, _num_local_freq, 1, 32, 2,
                         2, 0, device.getStream(cuda_stream_id), parameters, NULL);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("ERROR IN cuLaunchKernel: {}", errStr);
    }

    return record_end_event(gpu_frame_id);
}
