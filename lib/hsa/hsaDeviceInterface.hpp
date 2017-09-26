#ifndef GPU_HSA_DEVICE_INTERFACE_H
#define GPU_HSA_DEVICE_INTERFACE_H

#include <map>
#include <sys/mman.h>
#include <string>
#include <vector>
#include <map>
#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

#include "Config.hpp"
#include "buffer.h"

using std::vector;
using std::string;
using std::map;

// Parameters for the get_gpu_agent function
struct gpu_config_t {
  int gpu_id;
  hsa_agent_t *agent;
};

// Parameters for the get_device_memory_region function
struct gpu_mem_config_t {
  int gpu_id;
  hsa_amd_memory_pool_t *region;
};

// Store named set of gpu pointer(s) with uniform size
struct gpuMemoryBlock {
    vector<void*> gpu_pointers;
    int len;

    // Need to be able to release the hsa pointers
    ~gpuMemoryBlock();
};

class hsaDeviceInterface
{
public:
    hsaDeviceInterface(Config& config, int gpu_id);
    virtual ~hsaDeviceInterface();
    int get_gpu_id();
    int get_gpu_buffer_depth();

    // Get one of the gpu memory pointers with the given name and size = len at the given index
    // The size of the set is equal to gpu_buffer_depth, so index < gpu_buffer_depth
    // If a region with this name exists then it will just return an existing pointer
    // at the give index, if the region doesn't exist, then it creates
    // it with gpu_buffer_depth pointers of size len
    // NOTE: if accessing an existing named region then len must match the existing
    // length or the system will throw an assert.
    void *get_gpu_memory_array(const string &name, const int index, const int len);

    // Same as get_gpu_memory_array but gets just one gpu memory buffer
    // This can be used when internal memory is needed.
    // i.e. memory used for lookup tables that are the same between runs
    // or temporary buffers between kernels.
    // Should NOT be used for any memory that's copied between GPU and HOST memory.
    void *get_gpu_memory(const string &name, const int len);

    // Note, if precede_signal is 0, then we don't wait on any signal.
    // These functions should only be called once per command, and
    // the returned signal should be stored in signals[]
    // These functions are reentrant/thread-safe.
    hsa_signal_t async_copy_host_to_gpu(void *dst, void *src, int len,
                                        hsa_signal_t precede_signal,
                                        hsa_signal_t copy_signal);
    hsa_signal_t async_copy_gpu_to_host(void *dst, void *src, int len,
                                        hsa_signal_t precede_signal,
                                        hsa_signal_t copy_signal);

    // Not used for data copies, this should only be used for static tables, etc.
    // at start up.  Will block until all memory has been copied.
    // These functions are reentrant/thread-safe.
    void sync_copy_gpu_to_host(void *dst, void *src, int length);
    void sync_copy_host_to_gpu(void *dst, void *src, int length);

    hsa_agent_t get_gpu_agent();
    hsa_region_t get_kernarg_region();
    hsa_agent_t get_cpu_agent();
    hsa_queue_t * get_queue();
protected:

    Config &config;
    int gpu_id; // Internal GPU ID.

    // GPU HSA variables
    hsa_agent_t gpu_agent;
    hsa_amd_memory_pool_t global_region;
    hsa_region_t kernarg_region;

    // This might need to be more than one queue
    hsa_queue_t* queue;

    // GPU Information
    char agent_name[64];

    // CPU HSA variables
    hsa_agent_t cpu_agent;
    hsa_amd_memory_pool_t host_region;

    // Config variable
    int _gpu_buffer_depth;

private:

    map<string, gpuMemoryBlock> gpu_memory;

    static hsa_status_t get_gpu_agent(hsa_agent_t agent, void *data);
    static hsa_status_t get_cpu_agent(hsa_agent_t agent, void* data);
    static hsa_status_t get_kernarg_memory_region(hsa_region_t region, void* data);
    static hsa_status_t get_device_memory_region(hsa_amd_memory_pool_t region, void* data);
};

#endif // GPU_HSA_DEVICE_INTERFACE_H
