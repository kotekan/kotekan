#include "cudaCorrelatorKernel.cuh"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCorrelatorKernel);

cudaCorrelatorKernel::cudaCorrelatorKernel(Config& config, const string& unique_name,
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
}

cudaCorrelatorKernel::~cudaCorrelatorKernel() {}

using namespace nvcuda::wmma;
__global__
void corr(int *input, int *output, int ne, int nt, int nf)
{
    fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major> xr_matrix;
    fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major> xi_matrix;
    fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major> yr_matrix;
    fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major> yi_matrix;
    fragment<accumulator, 8, 8, 32, int> accr_matrix;
    fragment<accumulator, 8, 8, 32, int> acci_matrix;
    fill_fragment(accr_matrix, 0);
    fill_fragment(acci_matrix, 0);

    int n_blk = (ne/8)*(ne/8+1)/2;
    int blk_id = blockIdx.y;
    int f = blockIdx.z;

    //x,y coordinates of the upper blocks, cf OpenCL in Action p260
    int blk_x = blk_id;
    int blk_y = 0;
    int s = ne/8;
    while(blk_x >= s) {
        blk_x -= s--;
        blk_y++;
    }
    blk_x += blk_y;

    for (int t=0; t<nt/32; t++){
        load_matrix_sync(xi_matrix, input + t*4*ne*nf*2 + f*4*ne*2 + blk_x*2*8*4,       32);
        load_matrix_sync(xr_matrix, input + t*4*ne*nf*2 + f*4*ne*2 + blk_x*2*8*4 + 8*4, 32);
        load_matrix_sync(yi_matrix, input + t*4*ne*nf*2 + f*4*ne*2 + blk_y*2*8*4,       32);
        load_matrix_sync(yr_matrix, input + t*4*ne*nf*2 + f*4*ne*2 + blk_y*2*8*4 + 8*4, 32);

        mma_sync(accr_matrix, xi_matrix, yi_matrix, accr_matrix);
        mma_sync(accr_matrix, xr_matrix, yr_matrix, accr_matrix);
        mma_sync(acci_matrix, xi_matrix, yr_matrix, acci_matrix);
        mma_sync(acci_matrix, xr_matrix, yi_matrix, acci_matrix);
    }

    store_matrix_sync(output + f*n_blk*8*8*2 + 8*8*2*blk_id,       accr_matrix, 8, mem_col_major);
    store_matrix_sync(output + f*n_blk*8*8*2 + 8*8*2*blk_id + 8*8, acci_matrix, 8, mem_col_major);
}

cudaEvent_t cudaCorrelatorKernel::execute(int gpu_frame_id, cudaEvent_t pre_event) {
    pre_execute(gpu_frame_id);

    uint32_t input_frame_len = _num_elements * _num_local_freq * _samples_per_data_set;
    void *input_memory = device.get_gpu_memory_array("input", gpu_frame_id, input_frame_len);

    uint32_t output_len = _num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                          * _num_data_sets * sizeof(int32_t);
    void *output_memory = device.get_gpu_memory_array("output", gpu_frame_id, output_len);

    if (pre_event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(CUDA_COMPUTE_STREAM), pre_event, 0));
    CHECK_CUDA_ERROR(cudaEventCreate(&pre_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(pre_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    dim3 blk (8,4,1);
    dim3 grd (1,_num_blocks,_num_local_freq);
    corr<<<grd,blk,0,device.getStream(CUDA_COMPUTE_STREAM)>>>
        ((int*)input_memory, (int*)output_memory, _num_elements, _samples_per_data_set, _num_local_freq);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(post_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    return post_events[gpu_frame_id];
}
