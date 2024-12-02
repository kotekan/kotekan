#include <iostream>
#include <cuda_fp16.h>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------
//
// Some boilerplate, to use the same launchable kernel for f16 and i16.


struct local_transpose_f16
{
    using Dtype = __half2;
    
    static __device__ __forceinline__ void do_transpose(__half2 &x, __half2 &y)
    {
	__half2 xnew = __lows2half2(x, y);
	__half2 ynew = __highs2half2(x, y);
	x = xnew;
	y = ynew;
    }
};


struct local_transpose_i16
{
    using Dtype = unsigned int;
    
    static __device__ __forceinline__ void do_transpose(unsigned int &x, unsigned int &y)
    {
	unsigned int xnew = __byte_perm(x, y, 0x5410);
	unsigned int ynew = __byte_perm(x, y, 0x7632);
	x = xnew;
	y = ynew;
    }
};


// -------------------------------------------------------------------------------------------------


template<class T, typename D = typename T::Dtype>
__global__ void local_transpose_kernel(D *dst, const D *src, int niter)
{
    int it = threadIdx.x;
    int nt = blockDim.x;

    src += blockIdx.x * 8*nt;
    dst += blockIdx.x * 8*nt;
    
    D x0 = src[it];
    D x1 = src[it + nt];
    D x2 = src[it + 2*nt];
    D x3 = src[it + 3*nt];
    D x4 = src[it + 4*nt];
    D x5 = src[it + 5*nt];
    D x6 = src[it + 6*nt];
    D x7 = src[it + 7*nt];
	
    for (int i = 0; i < niter; i++) {
	T::do_transpose(x0, x1);
	T::do_transpose(x2, x3);
	T::do_transpose(x4, x5);
	T::do_transpose(x6, x7);
	
	T::do_transpose(x0, x2);
	T::do_transpose(x1, x3);
	T::do_transpose(x4, x6);
	T::do_transpose(x5, x7);
	
	T::do_transpose(x0, x4);
	T::do_transpose(x1, x5);
	T::do_transpose(x2, x6);
	T::do_transpose(x3, x7);
    }

    dst[it] = x0;
    dst[it + nt] = x1;
    dst[it + 2*nt] = x2;
    dst[it + 3*nt] = x3;
    dst[it + 4*nt] = x4;
    dst[it + 5*nt] = x5;
    dst[it + 6*nt] = x6;
    dst[it + 7*nt] = x7;
}


template<typename T>
void time_local_transpose_kernel(const char *name)
{
    using D = typename T::Dtype;
    static_assert(sizeof(D) == 4);
    
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int ncallbacks = 10;
    const int nstreams = 2;
    const int niter = 256 * 1024;
    const double tera_transposes_per_kernel = double(niter) * nblocks * nthreads_per_block * 12 / pow(2.,40.);

    const int ninner = nblocks * nthreads_per_block * 8;
    Array<D> dst({nstreams,ninner}, af_zero | af_gpu);
    Array<D> src({nstreams,ninner}, af_zero | af_gpu);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    D *d = dst.data + istream*ninner;
	    D *s = src.data + istream*ninner;

	    local_transpose_kernel<T> <<< nblocks, nthreads_per_block, 0, stream >>> (d, s, niter);
	    CUDA_PEEK(name);
	};

    CudaStreamPool pool(callback, ncallbacks, nstreams, name);
    pool.monitor_throughput("teratransposes / sec", tera_transposes_per_kernel);
    pool.run();
}


int main(int argc, char **argv)
{
    // Implements command-line usage: program [device].
    set_device_from_command_line(argc, argv);

    cout << "** FIXME: local_transpose_f16() timings are misleadingly optimistic! **" << endl;
    time_local_transpose_kernel<local_transpose_f16> ("local_transpose_f16");
    time_local_transpose_kernel<local_transpose_i16> ("local_transpose_i16");
    return 0;
}
