#include <iostream>
#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------


__global__ void shfl_xor_kernel(float *dst, const float *src, int niter)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    float x = src[s];
    
    for (int i = 0; i < niter; i++) {
	x += __shfl_xor_sync(0xffffffff, x, 0x1);
	x += __shfl_xor_sync(0xffffffff, x, 0x2);
	x += __shfl_xor_sync(0xffffffff, x, 0x4);
	x += __shfl_xor_sync(0xffffffff, x, 0x8);
    }

    dst[s] = x;
}


static void time_shfl_xor(int nblocks, int nthreads, int nstreams, int ncallbacks, int niter)
{
    int s = nblocks * nthreads;
    Array<float> dst_arr({nstreams,s}, af_zero | af_gpu);
    Array<float> src_arr({nstreams,s}, af_zero | af_gpu);

    // gigashuffles per callback
    double gsh = 4. * double(s) * double(niter) / pow(2,30.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    float *dst = dst_arr.data + istream * s;
	    float *src = src_arr.data + istream * s;
	    
	    shfl_xor_kernel <<<nblocks, nthreads>>> (dst, src, niter);

	    if (pool.num_callbacks == 0)
		return;
	    
	    cout << "    time_shfl_xor [" << pool.num_callbacks
		 << "]: avg time = " << pool.time_per_callback
		 << ", Gshuffles/sec = " << (gsh / pool.time_per_callback)
		 << endl;
	};

    CudaStreamPool pool(callback, ncallbacks, nstreams);
    pool.run();
}


// -------------------------------------------------------------------------------------------------


__global__ void reduce_add_kernel(int *dst, const int *src, int niter)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int x = src[s];
    
    for (int i = 0; i < niter; i++)
	x = __reduce_add_sync(0xffffffff, x);

    dst[s] = x;
}


static void time_reduce_add(int nblocks, int nthreads, int nstreams, int ncallbacks, int niter)
{
    int s = nblocks * nthreads;
    Array<int> dst_arr({nstreams,s}, af_zero | af_gpu);
    Array<int> src_arr({nstreams,s}, af_zero | af_gpu);

    // gigareduces per callback
    double gre = double(s) * double(niter) / pow(2,30.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    int *dst = dst_arr.data + istream * s;
	    int *src = src_arr.data + istream * s;
	    
	    reduce_add_kernel <<<nblocks, nthreads>>> (dst, src, niter);

	    if (pool.num_callbacks == 0)
		return;
	    
	    cout << "    time_reduce_add [" << pool.num_callbacks
		 << "]: avg time = " << pool.time_per_callback
		 << ", Greduces/sec = " << (gre / pool.time_per_callback)
		 << endl;
	};

    CudaStreamPool pool(callback, ncallbacks, nstreams);
    pool.run();
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    // (nblocks, nthreads, nstreams, ncallbacks, niter)
    time_shfl_xor(1000, 128, 2, 10, 1000000); 
    time_reduce_add(1000, 128, 2, 10, 3000000);
    return 0;
}
