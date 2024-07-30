#include <random>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iostream>

#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/string_utils.hpp"  // from_str()

using namespace std;


namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


runtime_error make_cuda_exception(cudaError_t xerr, const char *xstr, const char *file, int line)
{
    stringstream ss;
    
    ss << "CUDA error: " << xstr << " returned " << xerr
       << " (" << cudaGetErrorString(xerr) << ")"
       << " [" << file << ":" << line << "]";

    return runtime_error(ss.str());
}


void assign_kernel_dims(dim3 &nblocks, dim3 &nthreads, long nx, long ny, long nz, int threads_per_block, bool noisy)
{
    // Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
    static constexpr long max_xblocks = (1L << 31) - 1;
    static constexpr long max_yblocks = (1L << 16) - 1;
    static constexpr long max_zblocks = (1L << 16) - 1;
    static constexpr long max_zthreads = 64;

    if (threads_per_block < 32)
	throw runtime_error("assign_kernel_dims(): threads_per_block must be >= 32");
    if (threads_per_block > 1024)
	throw runtime_error("assign_kernel_dims(): threads_per_block must be <= 1024");
    if (threads_per_block & (threads_per_block-1))
	throw runtime_error("assign_kernel_dims(): threads_per_block must be a power of two");

    if (nx <= 0)
	throw runtime_error("assign_kernel_dims(): nx must be > 0");
    if (ny <= 0)
	throw runtime_error("assign_kernel_dims(): ny must be > 0");
    if (nz <= 0)
	throw runtime_error("assign_kernel_dims(): nz must be > 0");

    // Throughout this function, 'x' denotes log2(nthreads.x), and analogously for y.
    int xbest = 0;
    int ybest = 0;
    long nbest = 0;

    // Just brute-force loop over all choices of (x,y,z), rather than looking
    // for a fast algorithm since the number of possible choices is small.
    
    for (int x = 5; (1<<x) <= threads_per_block; x++) {
	long tx = 1 << x;  // nthreads.x
	long nx_pad = (nx+tx-1) & ~(tx-1);
	
	if (nx > (max_xblocks << x))
	    continue;
	
	for (int y = 0; (1<<(x+y)) <= threads_per_block; y++) {
	    long ty = 1 << y;                      // nthreads.y
	    long tz = threads_per_block >> (x+y);  // nthreads.z

	    if (tz > max_zthreads)
		continue;
	    if (ny > (max_yblocks << y))
		continue;
	    if (nz > ((max_zblocks * threads_per_block) >> (x+y)))
		continue;
	    
	    // A caller which sets ny=1 should always get nblocks.y == nthreads.y == 1 (and likewise for z).
	    if ((ny == 1) && (ty != 1))
		continue;
	    if ((nz == 1) && (tz != 1))
		continue;
				 
	    long ny_pad = (ny+ty-1) & ~(ty-1);
	    long nz_pad = (nz+tz-1) & ~(tz-1);
	    long npad = nx_pad * ny_pad * nz_pad;

	    if ((npad <= nbest) || (nbest == 0)) {
		xbest = x;
		ybest = y;
		nbest = npad;
	    }
	}
    }

    if (nbest == 0) {
	stringstream ss;
	ss << "assign_kernel_dims() failed: (ny,nz,nz)=(" << nx << "," << ny << "," << nz << ") is too large";
	throw runtime_error(ss.str());
    }

    nthreads.x = (1 << xbest);
    nthreads.y = (1 << ybest);
    nthreads.z = (threads_per_block >> (xbest + ybest));
    
    nblocks.x = (nx + nthreads.x - 1) >> xbest;
    nblocks.y = (ny + nthreads.y - 1) >> ybest;
    nblocks.z = (nz + nthreads.z - 1) / nthreads.z;

    if (noisy) {
	long n = nx * ny * nz;
	long npad = threads_per_block * nblocks.x * nblocks.y * nblocks.z;
	double overhead = double(npad-n) / double(n);
	
	cout << "assign_kernel_dims: (nx,ny,nz,T)=(" << nx << "," << ny << "," << nz << "," << threads_per_block
	     << "): nblocks=" << dim3_str(nblocks) << ", nthreads=" << dim3_str(nthreads) << ", overhead=" << overhead << endl;
    }
}


// Implements command-line usage: program [device].
void set_device_from_command_line(int argc, char **argv)
{
    int ndev = -1;
    CUDA_CALL(cudaGetDeviceCount(&ndev));

    if (ndev <= 0) {
	cerr << "No GPUs found! (cudaGetDeviceCount() returned zero";
	exit(1);
    }
    else if (argc == 2) {
	int dev = from_str<int> (argv[1]);
	if ((dev < 0) || (dev >= ndev)) {
	    cerr << "Invalid GPU (=" << dev << ") was specified (expected 0 <= gpu < " << ndev << ")\n";
	    exit(1);
	}

	CUDA_CALL(cudaSetDevice(dev));
	// Fall through to announce device
    }
    else if (argc != 1) {
	cerr << "Usage: " << argv[0] << " [device]" << endl;
	exit(2);
    }
    else if (ndev != 1) {
	cout << "Using default CUDA GPU (can override by specifying device on command line)\n";
	// Fall through to announce device
    }
    
    int dev = -1;
    CUDA_CALL(cudaGetDevice(&dev));

    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, dev));

    cout << "Using GPU=" << dev << ", name = " << prop.name << endl;
}


double get_sm_cycles_per_second(int device)
{
    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));

    // prop.clockRate is in kHz
    return 1.0e3 * double(prop.multiProcessorCount) * double(prop.clockRate);
}


} // namespace gputils
