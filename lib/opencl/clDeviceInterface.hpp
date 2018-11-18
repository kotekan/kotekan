#ifndef CL_DEVICE_INTERFACE_H
#define CL_DEVICE_INTERFACE_H

#include <map>
#include <sys/mman.h>
#include <vector>
#ifdef __APPLE__
    #include <OpenCL/cl_platform.h>
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl_platform.h>
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

using std::vector;
using std::string;
using std::map;

//check pagesize:
//getconf PAGESIZE
// result: 4096
#define PAGESIZE_MEM 4096

// This adjusts the number of queues used by the OpenCL runtime
// One queue is for data transfers to the GPU, one is for kernels,
// and one is for data transfers from the GPU to host memory.
// Unless you really know what you are doing, don't change this.
#define NUM_QUEUES 3

#include "fpga_header_functions.h"
#include "Config.hpp"
#include "buffer.h"
#include "kotekanLogging.hpp"

#ifdef WITH_OPENCL

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
#endif

char* oclGetOpenCLErrorCodeStr(cl_int input);
#define CHECK_CL_ERROR( err )                                       \
    if ( err ) {                                                    \
        internal_logging(LOG_ERR, "Error at %s:%d; Error type: %s", \
                __FILE__, __LINE__, oclGetOpenCLErrorCodeStr(err)); \
        std::abort();                                               \
    }
//WITH_OPENCL
#endif

// Store named set of gpu pointer(s) with uniform size
struct clMemoryBlock {
    vector<cl_mem> gpu_pointers;
    uint32_t len;

    // Need to be able to release the cl pointers
    ~clMemoryBlock();
};

class clDeviceInterface: public kotekanLogging
{
public:
    clDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_);
    ~clDeviceInterface();
    cl_mem get_device_freq_map(int32_t encoded_stream_id);

    cl_command_queue getQueue(int param_Dim);
    int getAlignedAccumulateLen() const;
    void prepareCommandQueue(bool enable_profiling);

    void release_events_for_buffer(int param_BufferID);
//    void deallocateResources();
    size_t get_opencl_resolution();
    cl_context &get_context();
    cl_device_id get_id();

    // Get one of the gpu memory pointers with the given name and size = len at the given index
    // The size of the set is equal to gpu_buffer_depth, so index < gpu_buffer_depth
    // If a region with this name exists then it will just return an existing pointer
    // at the give index, if the region doesn't exist, then it creates
    // it with gpu_buffer_depth pointers of size len
    // NOTE: if accessing an existing named region then len must match the existing
    // length or the system will throw an assert.
    cl_mem get_gpu_memory_array(const string &name, const uint32_t index, const uint32_t len);

    // Same as get_gpu_memory_array but gets just one gpu memory buffer
    // This can be used when internal memory is needed.
    // i.e. memory used for lookup tables that are the same between runs
    // or temporary buffers between kernels.
    // Should NOT be used for any memory that's copied between GPU and HOST memory.
    cl_mem get_gpu_memory(const string &name, const uint32_t len);


 protected:

    // Extra data
    Config &config;

    int gpu_id;

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue[NUM_QUEUES];

    // <streamID, freq_map>
    std::map<int32_t, cl_mem> device_freq_map;

    // Config variables
    uint32_t gpu_buffer_depth;

private:
    int num_local_freq;
    map<string, clMemoryBlock> gpu_memory;


};

#endif // CL_DEVICE_INTERFACE_H
