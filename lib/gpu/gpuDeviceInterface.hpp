#ifndef GPU_DEVICE_INTERFACE_H
#define GPU_DEVICE_INTERFACE_H

#include <map>
#include <vector>
#include "Config.hpp"
#include "buffer.h"
#include "kotekanLogging.hpp"

// Store named set of gpu pointer(s) with uniform size
struct gpuMemoryBlock {
    vector<void *> gpu_pointers;
    uint32_t len;
};

class gpuDeviceInterface: public kotekanLogging
{
public:
    gpuDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_);
    virtual ~gpuDeviceInterface();

    // Get one of the gpu memory pointers with the given name and size = len at the given index
    // The size of the set is equal to gpu_buffer_depth, so index < gpu_buffer_depth
    // If a region with this name exists then it will just return an existing pointer
    // at the give index, if the region doesn't exist, then it creates
    // it with gpu_buffer_depth pointers of size len
    // NOTE: if accessing an existing named region then len must match the existing
    // length or the system will throw an assert.
    void *get_gpu_memory_array(const string &name, const uint32_t index, const uint32_t len);

    // Same as get_gpu_memory_array but gets just one gpu memory buffer
    // This can be used when internal memory is needed.
    // i.e. memory used for lookup tables that are the same between runs
    // or temporary buffers between kernels.
    // Should NOT be used for any memory that's copied between GPU and HOST memory.
    void *get_gpu_memory(const string &name, const uint32_t len);

    // Can't do this in the destructor because only the derived classes know
    // how to free their memory.
    void cleanup_memory();

protected:
    virtual void* alloc_gpu_memory(int len) = 0;
    virtual void free_gpu_memory(void *) = 0;

    // Extra data
    Config &config;

    // Config variables
    int gpu_id;
    uint32_t gpu_buffer_depth;

private:
    std::map<string, gpuMemoryBlock> gpu_memory;
};

#endif // GPU_DEVICE_INTERFACE_H
