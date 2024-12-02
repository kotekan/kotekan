#include <sstream>
#include <iostream>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/device_mma.hpp"

// C++ WMMAs
#include <mma.h>

using namespace std;
using namespace nvcuda;
using namespace gputils;


// -------------------------------------------------------------------------------------------------
//
// float16 MMA


__device__ __half2 load_half2(const float *p)
{
    float2 a = *((float2 *) p);
    return __float22half2_rn(a);
}


__device__ void store_half2(float *p, __half2 x)
{
    float2 a = __half22float2(x);
    *((float2 *) p) = a;
}


// The Areg, Breg, Creg template arguments are the number of registers per thread
// needed to store the A,B,C matrices respectively.
//
// The 'asrc' array length is (Areg * nth * 2)
// The 'bsrc' array length is (Breg * nth * 2).
// The 'csrc' array length is (Creg * nth * 2).
//
// Here, nth = (nblocks * nthreads_per_block) is the total number of threads in the kernel.


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[]), int Areg, int Breg, int Creg>
__global__ void mma_f16_kernel(float *cdst, const float *asrc, const float *bsrc, int niter, int num_active_warps)
{
    int warpId = threadIdx.x >> 5;
    if (warpId >= num_active_warps)
	return;
    
    __half2 a[Areg];
    __half2 b[Breg];
    __half2 c[Creg];

    int ith = blockIdx.x * blockDim.x + threadIdx.x;
    int nth = gridDim.x * blockDim.x;

    #pragma unroll
    for (int r = 0; r < Areg; r++)
	a[r] = load_half2(asrc + r*nth*2 + ith*2);
    
    #pragma unroll
    for (int r = 0; r < Breg; r++)
	b[r] = load_half2(bsrc + r*nth*2 + ith*2);
    
    #pragma unroll
    for (int r = 0; r < Creg; r++)
	c[r] = __half2half2(0);
    
    for (int i = 0; i < niter; i++)
	F(c, a, b, c);
    
    #pragma unroll
    for (int r = 0; r < Creg; r++)
	store_half2(cdst + r*nth*2 + ith*2, c[r]);
}


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[]), int M, int N, int K, int S=1>
static void time_f16_mma(int niter, int num_active_warps=32)
{
    constexpr int Areg = (M*K*S) / 64;
    constexpr int Breg = (N*K*S) / 64;
    constexpr int Creg = (M*N*S) / 64;
    
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int ncallbacks = 10;
    const int nstreams = 2;

    int nth = nblocks * nthreads_per_block;
    Array<float> asrc({nstreams,Areg*nth*2}, af_zero | af_gpu);
    Array<float> bsrc({nstreams,Breg*nth*2}, af_zero | af_gpu);
    Array<float> csrc({nstreams,Creg*nth*2}, af_zero | af_gpu);

    double flops_per_mma = 2*M*N*K*S;
    double tflops_per_kernel = double(niter) * nblocks * num_active_warps * flops_per_mma / pow(2,40.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    float *a = asrc.data + istream*Areg*nth*2;
	    float *b = bsrc.data + istream*Breg*nth*2;
	    float *c = csrc.data + istream*Creg*nth*2;

	    mma_f16_kernel<F,Areg,Breg,Creg>
		<<< nblocks, nthreads_per_block, 0, stream >>>
		(c, a, b, niter, num_active_warps);

	    CUDA_PEEK("mma_f16_kernel");
	};

    stringstream ss;
    ss << "f16 (m=" << M << ", n=" << N << ", k=" << K << ")";
    
    if (num_active_warps < 32)
	ss << " **ACTIVE_WARPS=" << num_active_warps << "**";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, ss.str());
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}


// -------------------------------------------------------------------------------------------------
//
// int4/int8 MMA


// The Areg, Breg, Creg template arguments are the number of registers per thread
// needed to store the A,B,C matrices respectively.
//
// The 'asrc' array length is (Areg * nth)
// The 'bsrc' array length is (Breg * nth).
// The 'csrc' array length is (Creg * nth).
//
// Here, nth = (nblocks * nthreads_per_block) is the total number of threads in the kernel.


template<void (*F)(int[], const int[], const int[], const int[]), int Areg, int Breg, int Creg>
__global__ void mma_int_kernel(int *cdst, const int *asrc, const int *bsrc, int niter, int num_active_warps)
{
    int warpId = threadIdx.x >> 5;
    if (warpId >= num_active_warps)
	return;
    
    int a[Areg];
    int b[Breg];
    int c[Creg];
    
    int ith = blockIdx.x * blockDim.x + threadIdx.x;
    int nth = gridDim.x * blockDim.x;

    #pragma unroll
    for (int r = 0; r < Areg; r++)
	a[r] = asrc[r*nth + ith];
    
    #pragma unroll
    for (int r = 0; r < Breg; r++)
	a[r] = bsrc[r*nth + ith];

    #pragma unroll
    for (int r = 0; r < Creg; r++)
	c[r] = 0;

    for (int i = 0; i < niter; i++)
	F(c, a, b, c);
    
    #pragma unroll
    for (int r = 0; r < Creg; r++)
	cdst[r*nth + ith] = c[r];
}


template<void (*F)(int[], const int[], const int[], const int[]), int BitDepth, int M, int N, int K>
static void time_int_mma(int niter, int num_active_warps=32)
{
    constexpr int Areg = (M*K*BitDepth) / 1024;
    constexpr int Breg = (N*K*BitDepth) / 1024;
    constexpr int Creg = (M*N) / 32;
    
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int ncallbacks = 15;
    const int nstreams = 2;

    int nth = nblocks * nthreads_per_block;
    Array<int> asrc({nstreams,Areg*nth}, af_zero | af_gpu);
    Array<int> bsrc({nstreams,Breg*nth}, af_zero | af_gpu);
    Array<int> cdst({nstreams,Creg*nth}, af_zero | af_gpu);

    double flops_per_mma = 2*M*N*K;
    double tflops_per_kernel = double(niter) * nblocks * num_active_warps * flops_per_mma / pow(2,40.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    int *a = asrc.data + istream*Areg*nth;
	    int *b = bsrc.data + istream*Breg*nth;
	    int *c = cdst.data + istream*Creg*nth;

	    mma_int_kernel<F,Areg,Breg,Creg>
		<<< nblocks, nthreads_per_block, 0, stream >>>
		(c, a, b, niter, num_active_warps);

	    CUDA_PEEK("mma_int_kernel");
	};

    stringstream ss;
    ss << "int" << BitDepth << " (m=" << M << ", n=" << N << ", k=" << K << ")";

    if (num_active_warps < 32)
	ss << " **ACTIVE_WARPS=" << num_active_warps << "**";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, ss.str());
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}


// -------------------------------------------------------------------------------------------------


// The Areg, Breg, Creg template arguments are the number of registers per thread
// needed to store the A,B,C matrices respectively.
//
// The 'asrc' array length is (Areg * nth * 2)
// The 'bsrc' array length is (Breg * nth * 2).
// The 'csrc' array length is (Creg * nth * 2).
//
// Here, nth = (nblocks * nthreads_per_block) is the total number of threads in the kernel.


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[], unsigned int), int Areg, int Breg, int Creg>
__global__ void mma_sparse_f16_kernel(float *cdst, const float *asrc, const float *bsrc, int niter, int num_active_warps)
{
    int warpId = threadIdx.x >> 5;
    if (warpId >= num_active_warps)
	return;
    
    __half2 a[Areg];
    __half2 b[Breg];
    __half2 c[Creg];

    int ith = blockIdx.x * blockDim.x + threadIdx.x;
    int nth = gridDim.x * blockDim.x;

    #pragma unroll
    for (int r = 0; r < Areg; r++)
	a[r] = load_half2(asrc + r*nth*2 + ith*2);
    
    #pragma unroll
    for (int r = 0; r < Breg; r++)
	b[r] = load_half2(bsrc + r*nth*2 + ith*2);
    
    #pragma unroll
    for (int r = 0; r < Creg; r++)
	c[r] = __half2half2(0);
    
    for (int i = 0; i < niter; i++)
	F(c, a, b, c, 0);  // Note 0 here for sparse arg
    
    #pragma unroll
    for (int r = 0; r < Creg; r++)
	store_half2(cdst + r*nth*2 + ith*2, c[r]);
}


template<void (*F)(__half2[], const __half2[], const __half2[], const __half2[], unsigned int), int M, int N, int K>
static void time_sparse_f16_mma(int niter, int num_active_warps=32)
{
    constexpr int Areg = (M*K) / 128;  // note 128 here
    constexpr int Breg = (N*K) / 64;
    constexpr int Creg = (M*N) / 64;
    
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int ncallbacks = 10;
    const int nstreams = 2;

    int nth = nblocks * nthreads_per_block;
    Array<float> asrc({nstreams,Areg*nth*2}, af_zero | af_gpu);
    Array<float> bsrc({nstreams,Breg*nth*2}, af_zero | af_gpu);
    Array<float> csrc({nstreams,Creg*nth*2}, af_zero | af_gpu);

    double flops_per_mma = 2*M*N*K;   // nvidia's definition of sparse flops
    double tflops_per_kernel = double(niter) * nblocks * num_active_warps * flops_per_mma / pow(2,40.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    float *a = asrc.data + istream*Areg*nth*2;
	    float *b = bsrc.data + istream*Breg*nth*2;
	    float *c = csrc.data + istream*Creg*nth*2;

	    mma_sparse_f16_kernel<F,Areg,Breg,Creg>
		<<< nblocks, nthreads_per_block, 0, stream >>>
		(c, a, b, niter, num_active_warps);

	    CUDA_PEEK("sparse_mma_f16_kernel");
	};

    stringstream ss;
    ss << "sparse f16 (m=" << M << ", n=" << N << ", k=" << K << ")";
    
    if (num_active_warps < 32)
	ss << " **ACTIVE_WARPS=" << num_active_warps << "**";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, ss.str());
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}


// -------------------------------------------------------------------------------------------------
//
// C++ WMMA kernel


// The 'asrc' array length is (nwarps*32).
// The 'bsrc' array length is (nwarps*32).
// The 'csrc' array length is (nwarps*64).
//
// Here, nwarps = (nblocks * nwarps_per_block) is the total number of warps in the kernel.

__global__ void cpp_int4_kernel(int *cdst, const int *asrc, const int *bsrc, int niter, int num_active_warps)
{
    int warpId = threadIdx.x >> 5;
    int iwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    if (warpId >= num_active_warps)
	return;

    wmma::fragment<wmma::matrix_a, 8, 8, 32, wmma::experimental::precision::s4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, wmma::experimental::precision::s4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> c_frag;

    wmma::load_matrix_sync(a_frag, asrc + 32*iwarp, 32);
    wmma::load_matrix_sync(b_frag, bsrc + 32*iwarp, 32);
    wmma::fill_fragment(c_frag, 0);

    for (int i = 0; i < niter; i++)
	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   wmma::store_matrix_sync(cdst + 64*iwarp, c_frag, 8, wmma::mem_row_major);
}


static void time_cpp_int4_mma(int niter, int num_active_warps=32)
{
    const int nblocks = 82 * 84;
    const int nthreads_per_block = 1024;
    const int ncallbacks = 15;
    const int nstreams = 2;

    int nwarps = nblocks * (nthreads_per_block / 32);
    Array<int> asrc({nstreams,32*nwarps}, af_zero | af_gpu);
    Array<int> bsrc({nstreams,32*nwarps}, af_zero | af_gpu);
    Array<int> cdst({nstreams,64*nwarps}, af_zero | af_gpu);

    const int M = 8;
    const int N = 8;
    const int K = 32;
    double flops_per_mma = 2*M*N*K;
    double tflops_per_kernel = double(niter) * nblocks * num_active_warps * flops_per_mma / pow(2,40.);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    int *a = asrc.data + istream*32*nwarps;
	    int *b = bsrc.data + istream*32*nwarps;
	    int *c = cdst.data + istream*64*nwarps;

	    cpp_int4_kernel
		<<< nblocks, nthreads_per_block, 0, stream >>>
		(c, a, b, niter, num_active_warps);

	    CUDA_PEEK("mma_cpp_int4_kernel");
	};

    stringstream ss;
    ss << "C++ int4 (m=8, n=8, k=32)";

    if (num_active_warps < 32)
	ss << " **ACTIVE_WARPS=" << num_active_warps << "**";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, ss.str());
    pool.monitor_throughput("Tflops", tflops_per_kernel);
    pool.run();
}


// -------------------------------------------------------------------------------------------------


static void time_mmas(int num_active_warps=32)
{
    // float16
    time_f16_mma <mma_f16_m16_n8_k8, 16, 8, 8> (1024*1024/num_active_warps, num_active_warps);
    time_f16_mma <mma_f16_m16_n8_k16, 16, 8, 16> (512*1024/num_active_warps, num_active_warps);

    // int8
    time_int_mma <mma_s8_m8_n8_k16, 8, 8, 8, 16> (2048*1024/num_active_warps, num_active_warps);
    time_int_mma <mma_s8_m16_n8_k16, 8, 16, 8, 16> (1024*1024/num_active_warps, num_active_warps);
    time_int_mma <mma_s8_m16_n8_k32, 8, 16, 8, 32> (512*1024/num_active_warps, num_active_warps);

    // int4
    time_int_mma <mma_s4_m8_n8_k32, 4, 8, 8, 32> (2048*1024/num_active_warps, num_active_warps);
    time_int_mma <mma_s4_m16_n8_k32, 4, 16, 8, 32> (1024*1024/num_active_warps, num_active_warps);
    time_int_mma <mma_s4_m16_n8_k64, 4, 16, 8, 64> (512*1024/num_active_warps, num_active_warps);

    // int1
    time_int_mma <mma_b1_m8_n8_k128, 1, 8, 8, 128> (2048*1024/num_active_warps, num_active_warps);

    // C++ int4
    time_cpp_int4_mma(2048*1024/num_active_warps, num_active_warps);

    // Sparse f16
    time_sparse_f16_mma <mma_sp_f16_m16_n8_k16<0>, 16, 8, 16> (1024*1024/num_active_warps, num_active_warps);
    time_sparse_f16_mma <mma_sp_f16_m16_n8_k32<0>, 16, 8, 32> (512*1024/num_active_warps, num_active_warps);
    
    // The PTX ISA includes f16 m8n8k4 MMAs.
    // I tried generating wrappers for these, but timing showed that they were extremely slow.
    // I assume these MMAs are legacy instructions which are emulated on Ampere.
    // I left commented-out code here, and in ../generate_device_mma_hpp.py in case I ever want to revisit this.
    //
    // time_f16_mma <mma_f16_m8_n8_k4_rr, 8, 8, 4, 4> (1024*1024/num_active_warps, num_active_warps);
    // time_f16_mma <mma_f16_m8_n8_k4_rc, 8, 8, 4, 4> (1024*1024/num_active_warps, num_active_warps);
    // time_f16_mma <mma_f16_m8_n8_k4_cr, 8, 8, 4, 4> (1024*1024/num_active_warps, num_active_warps);
    // time_f16_mma <mma_f16_m8_n8_k4_cc, 8, 8, 4, 4> (1024*1024/num_active_warps, num_active_warps);
}


int main(int argc, char **argv)
{
    // Implements command-line usage: program [device].
    set_device_from_command_line(argc, argv);

    time_mmas(32);  // full occupancy (32 warps/SM)
    time_mmas(8);   // low occupancy (8 warps/SM)
    time_mmas(4);   // even lower occupancy (4 warps/SM)
    
    return 0;
}
