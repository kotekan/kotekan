#include "cudaShuffleAstron.cuh"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaShuffleAstron);

cudaShuffleAstron::cudaShuffleAstron(Config& config, const std::string& unique_name,
                                     bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "cudaShuffleRomein", "cudaShuffleRomein.cu") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _block_size = config.get<int>(unique_name, "block_size");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_ordered_voltage = config.get<std::string>(unique_name, "gpu_mem_ordered_voltage");

    set_command_type(gpuCommandType::KERNEL);
}

cudaShuffleAstron::~cudaShuffleAstron() {}

__global__ void shuffle_astron(int *input, int *output, int ne, int nt, int nf) {
    // Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_STATIONS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
    // NR_TIMES_PER_BLOCK	32 = (128 / (NR_BITS))

    // input[nt][nf][ne]
    // output[nf][nt/32][ne/2][2][32]

    /*
    //straight copy
    int E = blockIdx.x;
    int T = blockIdx.z;
    int f = blockIdx.y;
    int e = threadIdx.x + E*8;
    for (int t=0; t<1024; t++){
        int data_in = input[((T*1024 + t)*nf + f)*ne/4 + e];
        output[((T*1024 + t)*nf + f)*ne/4 + e] = data_in;
    }*/

    // read 32 times, 32 elements per warp
    // read 4 times, 4 elements per thread
    int E = blockIdx.x * 32 + threadIdx.x * 4;
    int T = blockIdx.y * 32 + threadIdx.y * 4;
    int F = blockIdx.z;

    int d[4], dd[4];
    //d[t], [e] inside words
    //in[nt,nf,ne]
    for (int t=0; t<4; t++)
        d[t] = input[((T + t) * nf + F) * ne / 4 + E / 4];
    dd[0] = ((d[0] & 0x000000ff)>> 0) | ((d[1] & 0x000000ff)<< 8) | ((d[2] & 0x000000ff)<<16) | ((d[3] & 0x000000ff)<<24);
    dd[1] = ((d[0] & 0x0000ff00)>> 8) | ((d[1] & 0x0000ff00)<< 0) | ((d[2] & 0x0000ff00)<< 8) | ((d[3] & 0x0000ff00)<<16);
    dd[2] = ((d[0] & 0x00ff0000)>>16) | ((d[1] & 0x00ff0000)>> 8) | ((d[2] & 0x00ff0000)>> 0) | ((d[3] & 0x00ff0000)<< 8);
    dd[3] = ((d[0] & 0xff000000)>>24) | ((d[1] & 0xff000000)>>16) | ((d[2] & 0xff000000)>> 8) | ((d[3] & 0xff000000)<< 0);
    //dd[e], [t] inside words
    //flip re, im; convert from offset encoded
    for (int e=0; e<4; e++)
        dd[e] = (((dd[e]&0xf0f0f0f0)>>4) | ((dd[e]&0x0f0f0f0f)<<4))^0x88888888;
    //out[nf,nt/32,ne,32]
    for (int e=0; e<4; e++)
        output[((F * nt/32 + blockIdx.y)*ne + E+e)*8 + threadIdx.y] = dd[e];
}

cudaEvent_t cudaShuffleAstron::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    void *input_memory = device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, input_frame_len);
    void *output_memory = device.get_gpu_memory(_gpu_mem_ordered_voltage, input_frame_len);

    if (pre_events[cuda_stream_id]) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(cuda_stream_id),
                                             pre_events[cuda_stream_id], 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(start_events[gpu_frame_id], device.getStream(cuda_stream_id)));

    dim3 blk (8,8,1);
    dim3 grd (_num_elements/32,_samples_per_data_set/32,_num_local_freq);
    shuffle_astron<<<grd,blk,0,device.getStream(cuda_stream_id)>>>
        ((int*)input_memory, (int*)output_memory, _num_elements, _samples_per_data_set, _num_local_freq);

    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventCreate(&end_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(end_events[gpu_frame_id], device.getStream(cuda_stream_id)));

    return end_events[gpu_frame_id];
}