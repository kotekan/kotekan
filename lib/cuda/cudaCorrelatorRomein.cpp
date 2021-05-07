#include "cudaCorrelatorRomein.hpp"
#include "math.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCorrelatorRomein);

cudaCorrelatorRomein::cudaCorrelatorRomein(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers, cudaDeviceInterface& device) :
        cudaCommand(config, unique_name, host_buffers, device, "correlate", "Correlator.cu") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    command_type = gpuCommandType::KERNEL;

    std::string stations = fmt::format("-DNR_STATIONS={:d}",_num_elements/2);
    std::string channels = fmt::format("-DNR_CHANNELS={:d}",_num_local_freq);
    std::string nsamples = fmt::format("-DNR_SAMPLES_PER_CHANNEL={:d}",_samples_per_data_set);
    std::string nstnpblk = fmt::format("-DNR_STATIONS_PER_BLOCK={:d}",64 /*48*/);
    const char *opts[] = {"-arch=compute_75",
                          "-std=c++14",
                          //"-code=sm_75",
                          "-lineinfo",
                          //"-src-in-ptx",
                          "-DNR_BITS=4",
                          stations.c_str(),
                          channels.c_str(),
                          nsamples.c_str(),
                          nstnpblk.c_str(),
                          "-DNR_POLARIZATIONS=2",
                          "-I/usr/local/cuda/include"
    };
    build(opts,10);
}

cudaCorrelatorRomein::~cudaCorrelatorRomein() {}

cudaEvent_t cudaCorrelatorRomein::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
//    void *input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);
    void *input_memory = device.get_gpu_memory("in_romein", input_frame_len);

    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                          * _num_data_sets * sizeof(int32_t);
    void *output_memory = device.get_gpu_memory_array("output", gpu_frame_id, output_len);

    if (pre_event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_COMPUTE_STREAM), pre_event, 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(pre_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    INFO("Starting...");
    CUresult err;
    void *parameters[] = { &output_memory, &input_memory };
//    int nblks = (_num_elements / 2 / 48) * (_num_elements / 2 / 48 + 1) / 2;
    int nblks = (_num_elements / 2 / 64) * (_num_elements / 2 / 64);
    err = cuLaunchKernel(corr_kernel,
                         nblks,_num_local_freq, 1,
                         32, 2, 2,
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
void cudaCorrelatorRomein::build(const char **opts, int nopts) {
    cudaCommand::build(opts, nopts);

    CUresult err;
    err = cuModuleGetFunction(&corr_kernel, module, "correlate");
    if (err != CUDA_SUCCESS){
        const char *errStr;
        cuGetErrorString(err, &errStr);
        INFO("ERROR IN cuModuleGetFunction for correlate: {}", errStr);
    }

    DEBUG2("BUILT CUDA KERNEL! Yay!");
}
