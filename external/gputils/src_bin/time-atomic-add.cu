#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/mem_utils.hpp"

using namespace std;
using namespace gputils;

using curand_state_t = curandStateXORWOW_t;
// using curand_state_t = curandStateMRG32k3a_t;   // around 4x slower than XORWOW


// -------------------------------------------------------------------------------------------------
//
// CurandStateArray helper class.
// FIXME move into main library some day.


struct CurandStateArray
{
    long nelts = 0;               // usually total number of threads in a kernel
    curand_state_t *data = 0;    // 1-d array of length nelts, on GPU

    // Launches kernel to init state, blocks until kernel is complete.
    // FIXME if moving this code into main library, launch on a stream instead.
    CurandStateArray(long nelts, ulong seed);

    // Enables reference counting, with call to cudaFree() when last reference is dropped.
    std::shared_ptr<void> ref;
};


__global__ void curand_init_kernel(curand_state_t *sp, ulong seed, long nelts)
{
    ulong t = ulong(blockIdx.x) * ulong(blockDim.x) + threadIdx.x;
    
    if (t < nelts)
	curand_init(seed, t, 0, sp+t);
}


CurandStateArray::CurandStateArray(long nelts_, ulong seed)
    : nelts(nelts_)
{
    assert(nelts > 0);
    assert((nelts % 32) == 0);
    assert(nelts <= 1024L * 1024L * 1024L);

    this->ref = _af_alloc(nelts * sizeof(curand_state_t), af_gpu);
    this->data = reinterpret_cast<curand_state_t *> (ref.get());

    long nblocks = (nelts + 127) >> 7;    
    curand_init_kernel<<<nblocks, 128>>> (data, seed, nelts);
    CUDA_PEEK("curand_init_kernel launch");
    CUDA_CALL(cudaDeviceSynchronize());
}


// -------------------------------------------------------------------------------------------------


__global__ void time_curand_kernel(uint *out, curand_state_t *state, int iterations_per_thread)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    curand_state_t st = state[t];
    
    uint x = 0;
    for (int i = 0; i < iterations_per_thread; i++)
	x ^= curand(&st);   // curand() produces a random uint32

    out[t] = x;
    state[t] = st;
}


static void time_curand()
{
    ulong seed = 32819837342L;
    int iterations_per_thread = 64*1024;
    int threads_per_block = 128;
    int nblocks = 16*1024;
    int nstreams = 2;
    int ncallbacks = 20;

    int threads_per_stream = nblocks * threads_per_block;
    int total_threads = nstreams * threads_per_stream;
    
    Array<uint> out({total_threads}, af_gpu);
    CurandStateArray state(total_threads, seed);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	time_curand_kernel<<< nblocks, threads_per_block, 0, stream >>>
	    (out.data + istream * threads_per_stream,
	     state.data + istream * threads_per_stream,
	     iterations_per_thread);
	
	CUDA_PEEK("time_curand_kernel launch");
    };

    CudaStreamPool sp(callback, ncallbacks, nstreams, "time_curand");
    sp.monitor_throughput("RNG throughput (Gsamples/s)", 1.0e-9 * iterations_per_thread * threads_per_stream);
    sp.run();
}


// -------------------------------------------------------------------------------------------------


// Warning: the meaning of 'nelts' depends on the value of 'sep_flag'!
//
//  - If sep_flag=false, then nelts = (total number of elements in global memory).
//  - If sep_flag=true, then nelts = (number of elements per threadblock).


template<typename T>
__global__ void global_atomic_add_kernel(T *p, curand_state_t *state, int iterations_per_thread, uint nelts, bool sep_flag)
{
    static constexpr uint ALL_LANES = 0xffffffffU;
    static constexpr T one = 1;
    
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    curand_state_t st = state[t];

    if (sep_flag)
	p += blockIdx.x * long(nelts);
    
    uint r = 0;
	
    for (int i = 0; i < iterations_per_thread; i++) {
	int il = (i & 31);
	
	if (il == 0) {
	    // Set r to a random number between 0 and nelts, divisible by 32.
	    r = curand(&st);   // curand() produces a random uint32
	    r = (r % nelts) & ~31;
	}

	// Index in 'p' array.
	int k = __shfl_sync(ALL_LANES,r,il) + (threadIdx.x & 31);
	atomicAdd(p+k, one);
    }

    state[t] = st;
}


template<typename T>
static void time_global_atomic_add(const string &name, int iterations_per_thread, uint nelts, bool sep_flag)
{
    ulong seed = 32819837342L;
    int threads_per_block = 128;
    int nblocks = 16*1024;
    int nstreams = 2;
    int ncallbacks = 20;

    int threads_per_stream = nblocks * threads_per_block;
    int total_threads = nstreams * threads_per_stream;

    long nb = sep_flag ? nblocks : 1;
    long nelts_per_stream = nb * nelts;
    long nelts_tot = nstreams * nelts_per_stream;
    assert(nelts_tot <= 1024L * 1024L * 1024L);
	
    Array<T> p({nelts_tot}, af_gpu | af_zero);
    CurandStateArray state(total_threads, seed);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	global_atomic_add_kernel<<< nblocks, threads_per_block, 0, stream >>>
	    (p.data + istream * nelts_per_stream,
	     state.data + istream * threads_per_stream,
	     iterations_per_thread,
	     nelts,
	     sep_flag);
	
	CUDA_PEEK("time_curand_kernel launch");
    };

    CudaStreamPool sp(callback, ncallbacks, nstreams, name);
    sp.monitor_throughput("Bandwidth (GB/s)", 2.0e-9 * iterations_per_thread * threads_per_stream * sizeof(T));
    sp.run();
}


static void time_global_atomic_add()
{
    // (name, iterations_per_thread, nelts, sep_flag)
    time_global_atomic_add<float> ("bigpool-fp32", 2*1024, 128*1024*1024, false);
    time_global_atomic_add<double> ("bigpool-fp64", 2*1024, 128*1024*1024, false);
    
    // (name, iterations_per_thread, nelts, sep_flag)
    time_global_atomic_add<float> ("tinypool-fp32", 2*1024, 8*1024, false);
    time_global_atomic_add<double> ("tinypool-fp64", 2*1024, 8*1024, false);
    
    // (name, iterations_per_thread, nelts, sep_flag)
    time_global_atomic_add<float> ("tinypool-sep-fp32", 2*1024, 1024, true);
    time_global_atomic_add<double> ("tinypool-sep-fp64", 2*1024, 1024, true);
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    time_curand();
    time_global_atomic_add();
    return 0;
}
