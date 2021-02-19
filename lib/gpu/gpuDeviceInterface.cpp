#include "gpuDeviceInterface.hpp"

#include <algorithm> // for max
#include <assert.h>  // for assert
#include <utility>   // for pair

using kotekan::Config;

gpuDeviceInterface::gpuDeviceInterface(Config& config_, int32_t gpu_id_, int gpu_buffer_depth_) :
    config(config_),
    gpu_id(gpu_id_),
    gpu_buffer_depth(gpu_buffer_depth_) {}

gpuDeviceInterface::~gpuDeviceInterface() {}

void gpuDeviceInterface::cleanup_memory() {
    for (auto it = gpu_memory.begin(); it != gpu_memory.end(); it++) {
        for (void* mem : it->second.gpu_pointers) {
            free_gpu_memory(mem);
        }
    }
}

void* gpuDeviceInterface::get_gpu_memory(const std::string& name, const uint32_t len) {

    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        void* ptr = alloc_gpu_memory(len);
        INFO("Allocating GPU[{:d}] memory: {:s}, len: {:d}, ptr: {:p}", gpu_id, name, len, ptr);
        gpu_memory[name].len = len;
        gpu_memory[name].gpu_pointers.push_back(ptr);
    }
    // The size must match what has already been allocated.
    assert(len == gpu_memory[name].len);
    assert(gpu_memory[name].gpu_pointers.size() == 1);

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[0];
}

void* gpuDeviceInterface::get_gpu_memory_array(const std::string& name, const uint32_t index,
                                               const uint32_t len) {
    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        for (uint32_t i = 0; i < gpu_buffer_depth; ++i) {
            void* ptr = alloc_gpu_memory(len);
            INFO("Allocating GPU[{:d}] memory: {:s}, len: {:d}, ptr: {:p}", gpu_id, name, len, ptr);
            gpu_memory[name].len = len;
            gpu_memory[name].gpu_pointers.push_back(ptr);
        }
    }
    // The size must match what has already been allocated.
    assert(len == gpu_memory[name].len);
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].gpu_pointers.size());

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[index];
}
