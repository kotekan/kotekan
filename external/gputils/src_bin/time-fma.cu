#include <sstream>
#include <iostream>
#include <cuda_fp16.h>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"

using namespace std;
using namespace gputils;


// -------------------------------------------------------------------------------------------------


template<typename T>
__global__ void fma_kernel(T *dst, const T *src, int niter)
{
    static_assert(sizeof(T) == 4);
    
    int ith = blockIdx.x * blockDim.x + threadIdx.x;
    int nth = gridDim.x * blockDim.x;
    
    T a = src[ith];
    T b = src[ith+nth];
    T c = src[ith+2*nth];
    T d = src[ith+3*nth];
    
    for (int i = 0; i < niter; i++) {
#if 1
	// Always fast
	a += c*c;
	b += d*d;
	c += a*a;
	d += b*b;
#else
	// Slow for fp32, but not fp16 (register bank conflicts?)
	a += b*c;
	b += c*d;
	c += d*a;
	d += a*b;
#endif
    }

    dst[ith] = a+b+c+d;
}


__global__ void hcmadd_kernel(__half2 *dst, const __half2 *src, int niter)
{
    int ith = blockIdx.x * blockDim.x + threadIdx.x;
    int nth = gridDim.x * blockDim.x;
    
    __half2 a = src[ith];
    __half2 b = src[ith+nth];
    __half2 c = src[ith+2*nth];
    __half2 d = src[ith+3*nth];
    
    for (int i = 0; i < niter; i++) {
#if 1
	// Faster
	a = __hcmadd(c, c, a);
	b = __hcmadd(d, d, b);
	c = __hcmadd(a, a, c);
	d = __hcmadd(b, b, d);
#else
	// Slower (register bank conflicts?)
	a = __hcmadd(b, c, a);
	b = __hcmadd(c, d, b);
	c = __hcmadd(d, a, c);
	d = __hcmadd(a, b, d);
#endif
    }

    dst[ith] = a+b+c+d;
}


// -------------------------------------------------------------------------------------------------


template<typename T, void (*F)(T *, const T *, int)>
static void time_kernel(const char *name, int flops_per_iteration)
{
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int ncallbacks = 10;
    const int nstreams = 2;
    const int niter = 4096 * 1024 / flops_per_iteration;
    const int nth = nblocks * nthreads_per_block;
    const double tflops_per_kernel = nth * double(niter) * flops_per_iteration / pow(2,40.);

    static_assert(sizeof(T) == 4);
    Array<int> dst({nstreams,nth}, af_zero | af_gpu);
    Array<int> src({nstreams,4*nth}, af_zero | af_gpu);
    
    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    T *d = (T *) (dst.data + istream*nth);
	    T *s = (T *) (src.data + istream*4*nth);
	    
	    F <<< nblocks, nthreads_per_block, 0, stream >>> (d,s,niter);
	    CUDA_PEEK(name);
	};

    
    CudaStreamPool pool(callback, ncallbacks, nstreams, name);
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}


int main(int argc, char **argv)
{
    // Implements command-line usage: program [device].
    set_device_from_command_line(argc, argv);

    time_kernel<float, fma_kernel<float>> ("fp32_fma", 4*2);
    time_kernel<__half2, fma_kernel<__half2>> ("fp16_fma", 4*4);
    time_kernel<__half2, hcmadd_kernel> ("hcmadd", 4*8);
    
    return 0;
}
	     
