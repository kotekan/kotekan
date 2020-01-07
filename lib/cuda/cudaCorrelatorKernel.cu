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

#define TILE_COARSE_SIZE 2
#define TENSOR_I 8 //samples
#define TENSOR_T 32 //samples
#define TIMES_PER_WORD 8
#define TENSOR_TWORDS 4
#define TENSOR_WORDS 32 //(TENSOR_I * TENSOR_T / TIMES_PER_WORD) //words
#define TILE_SIZE (TENSOR_I * TILE_COARSE_SIZE)
#define TIME_PRELOAD 8

__global__ void corr(int *input, int *output, const int ne, const int nt, const int nf)
{
    //[I][t][d]
//    __shared__ int xr[TILE_COARSE_SIZE][TILE_COARSE_SIZE][TENSOR_WORDS], xi[TILE_COARSE_SIZE][TILE_COARSE_SIZE][TENSOR_WORDS],
//                   yi[TILE_COARSE_SIZE][TILE_COARSE_SIZE][TENSOR_WORDS], yr[TILE_COARSE_SIZE][TILE_COARSE_SIZE][TENSOR_WORDS];
    __shared__ int d[4*TIME_PRELOAD*TILE_COARSE_SIZE*(0+TENSOR_WORDS)];
    //Y is data type, time_preload is a loop, X is the element id

    fragment<matrix_a, 8,8,32, experimental::precision::s4, row_major> xr_matrix;
    fragment<matrix_a, 8,8,32, experimental::precision::s4, row_major> xi_matrix;
    fragment<matrix_b, 8,8,32, experimental::precision::s4, col_major> yr_matrix;
    fragment<matrix_b, 8,8,32, experimental::precision::s4, col_major> yi_matrix;
//    fragment<accumulator, 8,8,32, int> accii_matrix;
    fragment<accumulator, 8,8,32, int> accrr_matrix;
//    fragment<accumulator, 8,8,32, int> accir_matrix;
    fragment<accumulator, 8,8,32, int> accri_matrix;
//    fill_fragment(accii_matrix, 0);
    fill_fragment(accrr_matrix, 0);
//    fill_fragment(accir_matrix, 0);
    fill_fragment(accri_matrix, 0);

    int n_tile = (ne/TILE_SIZE)*(ne/TILE_SIZE+1)/2;
    int tile_id = blockIdx.x;
    int f = blockIdx.y;

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
    int offset[4] = {tile_x*2, tile_x*2+1, tile_y*2, tile_y*2+1};

    for (int T=0; T<nt/TENSOR_T; T+=TIME_PRELOAD){
        //each tile does 4x4 of threads, each doing 8x8 submatrix
        //read 4 timesteps (id'd by Y) into shared memory:
/*
        xi[Y][X][l] = input[((T+Y)*nf+f)*TENSOR_TWORDS*ne*2 + ((tile_x*2 + 0)*TILE_COARSE_SIZE + X)*TENSOR_WORDS + l];
        xr[Y][X][l] = input[((T+Y)*nf+f)*TENSOR_TWORDS*ne*2 + ((tile_x*2 + 1)*TILE_COARSE_SIZE + X)*TENSOR_WORDS + l];
        yi[Y][X][l] = input[((T+Y)*nf+f)*TENSOR_TWORDS*ne*2 + ((tile_y*2 + 0)*TILE_COARSE_SIZE + X)*TENSOR_WORDS + l];
        yr[Y][X][l] = input[((T+Y)*nf+f)*TENSOR_TWORDS*ne*2 + ((tile_y*2 + 1)*TILE_COARSE_SIZE + X)*TENSOR_WORDS + l];
*/
        for (int t=0; t<TIME_PRELOAD; t++){
            d[(((2*Y+0)*TIME_PRELOAD+t)*TILE_COARSE_SIZE + X)*(0+TENSOR_WORDS) + l] = 
                input[((T+t)*nf+f)*TENSOR_TWORDS*ne*2 + (offset[2*Y+0]*TILE_COARSE_SIZE + X)*TENSOR_WORDS + l];
            d[(((2*Y+1)*TIME_PRELOAD+t)*TILE_COARSE_SIZE + X)*(0+TENSOR_WORDS) + l] = 
                input[((T+t)*nf+f)*TENSOR_TWORDS*ne*2 + (offset[2*Y+1]*TILE_COARSE_SIZE + X)*TENSOR_WORDS + l];
        }

        __syncthreads();
/*
        for (int t=0; t<TILE_COARSE_SIZE; t++){
            load_matrix_sync(xi_matrix, xi[(X+Y+t)%4][X], TENSOR_T);
            load_matrix_sync(xr_matrix, xr[(X+Y+t)%4][X], TENSOR_T);
            load_matrix_sync(yi_matrix, yi[(X+Y+t)%4][Y], TENSOR_T);
            load_matrix_sync(yr_matrix, yr[(X+Y+t)%4][Y], TENSOR_T);
*/
        for (int t=0; t<TIME_PRELOAD; t++){
            load_matrix_sync(xi_matrix, &d[((0*TIME_PRELOAD+t)*TILE_COARSE_SIZE + X)*(0+TENSOR_WORDS)], TENSOR_T);
            load_matrix_sync(xr_matrix, &d[((1*TIME_PRELOAD+t)*TILE_COARSE_SIZE + X)*(0+TENSOR_WORDS)], TENSOR_T);
            load_matrix_sync(yi_matrix, &d[((2*TIME_PRELOAD+t)*TILE_COARSE_SIZE + Y)*(0+TENSOR_WORDS)], TENSOR_T);
            load_matrix_sync(yr_matrix, &d[((3*TIME_PRELOAD+t)*TILE_COARSE_SIZE + Y)*(0+TENSOR_WORDS)], TENSOR_T);

            mma_sync(accrr_matrix, xi_matrix, yi_matrix, accrr_matrix);
            mma_sync(accrr_matrix, xr_matrix, yr_matrix, accrr_matrix);
            mma_sync(accri_matrix, xr_matrix, yi_matrix, accri_matrix);
            mma_sync(accri_matrix, xi_matrix, yr_matrix, accri_matrix);
        }
        __syncthreads();
    }

    store_matrix_sync(output + ((f*n_tile+tile_id)*2+0)*TILE_SIZE*TILE_SIZE + (Y*TILE_SIZE+X)*TENSOR_I,
                      accrr_matrix, TILE_SIZE, mem_col_major);
    store_matrix_sync(output + ((f*n_tile+tile_id)*2+1)*TILE_SIZE*TILE_SIZE + (Y*TILE_SIZE+X)*TENSOR_I,
                      accri_matrix, TILE_SIZE, mem_col_major);
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
    dim3 blk (32,TILE_COARSE_SIZE,TILE_COARSE_SIZE);
    dim3 grd (_num_blocks,_num_local_freq);
    corr<<<grd,blk,0,device.getStream(CUDA_COMPUTE_STREAM)>>>
        ((int*)input_memory, (int*)output_memory, _num_elements, _samples_per_data_set, _num_local_freq);
/*  dim3 blk (32,1,1);
    dim3 grd (1,_num_blocks,_num_local_freq);
    basic_corr<<<grd,blk,0,device.getStream(CUDA_COMPUTE_STREAM)>>>
        ((int*)input_memory, (int*)output_memory, _num_elements, _samples_per_data_set, _num_local_freq);*/
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventCreate(&post_events[gpu_frame_id]));
    CHECK_CUDA_ERROR(cudaEventRecord(post_events[gpu_frame_id], device.getStream(CUDA_COMPUTE_STREAM)));

    return post_events[gpu_frame_id];
}
