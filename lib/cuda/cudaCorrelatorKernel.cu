#include "cudaCorrelatorKernel.cuh"
#include "math.h"
#include "mma.h"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCorrelatorKernel);

cudaCorrelatorKernel::cudaCorrelatorKernel(Config& config, const std::string& unique_name,
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
__global__ void basic_corr(int *input, int *output, int ne, int nt, int nf)
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

#define TILE_COARSE_X 4
#define TILE_COARSE_Y 2
#define TENSOR_I 8 //samples
#define TENSOR_T 32 //samples
#define TENSOR_WORDS 32 //32b ints per tensor
#define TILE_SIZE 64
#define TIME_PRELOAD 1

#define ALIGN_OFFSET 0

__global__
void corr(int *input, int *output, const int ne, const int nt, const int nf)
{
    __shared__ int d[TIME_PRELOAD][TENSOR_WORDS * (TILE_SIZE/TENSOR_I) * 2 + // X inputs
                     ALIGN_OFFSET+ TENSOR_WORDS * (TILE_SIZE/TENSOR_I) * 2]; // Y inputs

    fragment<matrix_a, 8,8,32, experimental::precision::s4, row_major> xr_matrix[2];
    fragment<matrix_a, 8,8,32, experimental::precision::s4, row_major> xi_matrix[2];
    fragment<matrix_b, 8,8,32, experimental::precision::s4, col_major> yr_matrix[4];
    fragment<matrix_b, 8,8,32, experimental::precision::s4, col_major> yi_matrix[4];
    fragment<accumulator, 8,8,32, int> accrr_matrix[2][4];
    fragment<accumulator, 8,8,32, int> accri_matrix[2][4];

    for (int x=0; x<2; x++) {
        for (int y=0; y<4; y++) {
            fill_fragment(accrr_matrix[x][y], 0);
            fill_fragment(accri_matrix[x][y], 0);
        }
    }

    //int n_tile = (ne/TILE_SIZE)*(ne/TILE_SIZE+1)/2;
    int tile_id = blockIdx.x;
    //int f = blockIdx.y;

    //x,y coordinates of the upper tiles, cf OpenCL in Action p260
    int tile_x = tile_id;
    int tile_y = 0;
    int s = ne/TILE_SIZE;
    while(tile_x >= s) {
        tile_x -= s--;
        tile_y++;
    }
    tile_x += tile_y;

    int l=threadIdx.x;
    int X=threadIdx.y;
    int Y=threadIdx.z;

    int xoff = tile_x * TILE_SIZE * TENSOR_WORDS / TENSOR_I * 2;
    int yoff = tile_y * TILE_SIZE * TENSOR_WORDS / TENSOR_I * 2;
    for (int T=0; T<nt/TENSOR_T; T+=TIME_PRELOAD) {

        for (int t=0; t<TIME_PRELOAD; t++) {
            int4 *src_addr = (int4 *) &input[(T + t) * ne * 4 * 2 + (X * 32 + l) * 4 + (Y ? yoff : xoff)];
            int4 *dst_addr = (int4 *) &d[t][(X * 32 + l) * 4 + Y * (TENSOR_WORDS * 8 * 2 + ALIGN_OFFSET)];
            *dst_addr = *src_addr;
        }
        __syncthreads();

        for (int t=0; t<TIME_PRELOAD; t++) {
            for (int x=0; x<2; x++) {
                load_matrix_sync(xi_matrix[x], &d[t][((X*2+x) * 2 + 0) * TENSOR_WORDS], TENSOR_T);
                load_matrix_sync(xr_matrix[x], &d[t][((X*2+x) * 2 + 1) * TENSOR_WORDS], TENSOR_T);
            }
            for (int y=0; y<4; y++) {
                load_matrix_sync(yi_matrix[y], &d[t][((Y*4+y) * 2 + 0) * TENSOR_WORDS + TENSOR_WORDS*8*2 + ALIGN_OFFSET], TENSOR_T);
                load_matrix_sync(yr_matrix[y], &d[t][((Y*4+y) * 2 + 1) * TENSOR_WORDS + TENSOR_WORDS*8*2 + ALIGN_OFFSET], TENSOR_T);
            }
            for (int x=0; x<2; x++) {
                for (int y=0; y<4; y++) {
                    mma_sync(accrr_matrix[x][y], xi_matrix[x], yi_matrix[y], accrr_matrix[x][y]);
                    mma_sync(accrr_matrix[x][y], xr_matrix[x], yr_matrix[y], accrr_matrix[x][y]);
                    mma_sync(accri_matrix[x][y], xr_matrix[x], yi_matrix[y], accri_matrix[x][y]);
                    mma_sync(accri_matrix[x][y], xi_matrix[x], yr_matrix[y], accri_matrix[x][y]);
                }
            }
        }
        __syncthreads();
    }

    for (int x=0; x<2; x++) {
        for (int y=0; y<4; y++) {
            store_matrix_sync(output
                              + (tile_id*2+0)*TILE_SIZE*TILE_SIZE
                              + ((Y*4+y)*TILE_SIZE+(X*2+x))*TENSOR_I,
                              accrr_matrix[x][y], TILE_SIZE, mem_col_major);
            store_matrix_sync(output
                              + (tile_id*2+1)*TILE_SIZE*TILE_SIZE
                              + ((Y*4+y)*TILE_SIZE+(X*2+x))*TENSOR_I,
                              accri_matrix[x][y], TILE_SIZE, mem_col_major);
        }
    }

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

    CHECK_CUDA_ERROR(cudaFuncSetAttribute (corr, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    dim3 blk (32,4,2);
    dim3 grd (_num_blocks,_num_local_freq);
    corr<<<grd,blk,0,device.getStream(CUDA_COMPUTE_STREAM)>>>
        ((int*)input_memory, (int*)output_memory, _num_elements, _samples_per_data_set, _num_local_freq);

    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(post_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    return post_events[gpu_frame_id];
}
