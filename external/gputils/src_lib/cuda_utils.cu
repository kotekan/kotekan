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
