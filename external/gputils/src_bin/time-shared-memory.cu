#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"

#include <iostream>

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------
//
// 'p' should be a zeroed array of length (threads/block) * (blocks/kernel).
// Kernels should be launched with shmem_nbytes = 4 * (threads/block).


template<bool Read, bool Write>
__global__ void shmem_read_write_kernel(int *p, int niter)
{
    constexpr int N = (Read && Write) ? 2 : 1;
    extern __shared__ int shmem[];

    p += threadIdx.x + (blockIdx.x * blockDim.x);
    int x = *p;
    
    int s = threadIdx.x;
    shmem[s] = x;
    
    for (int i = 2; i < niter; i += N) {
	int y = Read ? shmem[s] : (x >> 1);
	x ^= y;
	
	if (Write)
	    shmem[s] = x;
	
	s ^= x;
    }

    *p = shmem[s];
}


template<bool Read, bool Write>
static void time_kernel(const char *name)
{
    const int niter = 8 * 1024;
    const int threads_per_block = 16 * 32;
    const int blocks_per_kernel = 32 * 1024;
    const int nstreams = 1;
    const int nkernels = 10;

    const double sm_cycles_per_second = get_sm_cycles_per_second();
    const double instructions_per_kernel = double(niter) * double(threads_per_block/32) * double(blocks_per_kernel);
    const double shmem_tb_per_kernel = 1.0e-12 * 128.0 * instructions_per_kernel;
    const int shmem_nbytes = 4 * threads_per_block;

    Array<int> arr({nstreams, blocks_per_kernel * threads_per_block }, af_gpu | af_zero);
    
    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	int *gmem = arr.data + (istream * arr.shape[1]);
	
	shmem_read_write_kernel<Read, Write>
	    <<< blocks_per_kernel, threads_per_block, shmem_nbytes, stream >>>
	    (gmem, niter);
	
	CUDA_PEEK(name);
    };

    CudaStreamPool sp(callback, nkernels, nstreams, name);
    sp.monitor_throughput("Shared memory BW (TB/s)", shmem_tb_per_kernel);
    sp.monitor_time("Clock cycles", instructions_per_kernel / sm_cycles_per_second);
    sp.run();
}

    
int main(int argc, char **argv)
{
    // Implements command-line usage: program [device].
    set_device_from_command_line(argc, argv);
    
    time_kernel<true,false> ("Read shared memory");
    time_kernel<false,true> ("Write shared memory");
    time_kernel<true,true> ("Read/write shared memory");
    
    return 0;
}
