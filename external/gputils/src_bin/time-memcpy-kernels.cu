#include <sstream>
#include <iostream>

#include "../include/gputils/Array.hpp"
#include "../include/gputils/CudaStreamPool.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/memcpy_kernels.hpp"

using namespace std;
using namespace gputils;


static void time_memcpy(long nbytes, int ninner, int nouter, int nstreams=1)
{
    Array<char> adst({nstreams,nbytes}, af_zero | af_gpu);
    Array<char> asrc({nstreams,nbytes}, af_zero | af_gpu);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	char *d = adst.data + istream*nbytes;
	char *s = asrc.data + istream*nbytes;

	for (int i = 0; i < ninner; i++)
	    launch_memcpy_kernel(d, s, nbytes, stream);
	
	CUDA_PEEK("launch_memcpy_kernel");
    };

    stringstream ss;
    ss << "memcpy(nbytes=" << nbytes << ")";

    double gb_per_kernel = 2.0e-9 * ninner * nbytes;
    CudaStreamPool pool(callback, nouter, nstreams, ss.str());
    pool.monitor_throughput("GB/s", gb_per_kernel);
    pool.run();
}


static void time_memcpy_2d(long dpitch, long spitch, long width, long height, int ninner, int nouter, int nstreams=1)
{
    long dst_nbytes = height * dpitch;
    long src_nbytes = height * spitch;
    
    Array<char> adst({nstreams,dst_nbytes}, af_zero | af_gpu);
    Array<char> asrc({nstreams,src_nbytes}, af_zero | af_gpu);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	char *d = adst.data + istream * dst_nbytes;
	char *s = asrc.data + istream * src_nbytes;

	for (int i = 0; i < ninner; i++)
	    launch_memcpy_2d_kernel(d, dpitch, s, spitch, width, height, stream);
	
	CUDA_PEEK("launch_memcpy_2d_kernel");
    };

    stringstream ss;
    ss << "memcpy_2d(dpitch=" << dpitch << ", spitch=" << spitch << ", width=" << width << ", height=" << height << ")";

    double gb_per_kernel = 2.0e-9 * ninner * width * height;
    CudaStreamPool pool(callback, nouter, nstreams, ss.str());
    pool.monitor_throughput("GB/s", gb_per_kernel);
    pool.run();
}


int main(int argc, char **argv)
{
    long gb4 = 4L * 1024L * 1024L * 1024L;
    
    time_memcpy(gb4, 20, 10);

    // (dpitch, spitch, width, height)
    time_memcpy_2d(65536+128, 65536+1024, 65536, 65536, 20, 10);
    time_memcpy_2d(128, 128, 128, 32L * 1024L * 1024L, 20, 10);
    time_memcpy_2d(256, 256, 128, 32L * 1024L * 1024L, 20, 10);
    time_memcpy_2d(gb4, gb4, gb4, 1, 20, 10);
    
    return 0;
}
