#include "gpuDeviceInterface.hpp"

#include "fmt.hpp"

#include <algorithm> // for max
#include <assert.h>  // for assert
#include <utility>   // for pair

using kotekan::Config;

gpuDeviceInterface::gpuDeviceInterface(Config& config, const std::string& unique_name,
                                       int32_t gpu_id, int gpu_buffer_depth) :
    config(config),
    unique_name(unique_name), gpu_id(gpu_id), gpu_buffer_depth(gpu_buffer_depth) {}

gpuDeviceInterface::~gpuDeviceInterface() {}

void gpuDeviceInterface::cleanup_memory() {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    for (auto it = gpu_memory.begin(); it != gpu_memory.end(); it++) {
        for (void* mem : it->second.gpu_pointers_to_free) {
            if (mem)
                free_gpu_memory(mem);
        }
    }
}

void* gpuDeviceInterface::get_gpu_memory(const std::string& name, const size_t len) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);

    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        void* ptr = alloc_gpu_memory(len);
        INFO("Allocating GPU[{:d}] memory: {:s}, len: {:d}, ptr: {:p}", gpu_id, name, len, ptr);
        gpu_memory[name].len = len;
        gpu_memory[name].view_source = "";
        gpu_memory[name].gpu_pointers.push_back(ptr);
        gpu_memory[name].gpu_pointers_to_free.push_back(ptr);
    }
    // The size must match what has already been allocated.
    assert(len == gpu_memory[name].len);
    assert(gpu_memory[name].gpu_pointers.size() == 1);

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[0];
}

void* gpuDeviceInterface::get_gpu_memory_array(const std::string& name, const uint32_t index,
                                               const size_t len) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    // Check if the memory isn't yet allocated
    if (gpu_memory.count(name) == 0) {
        // Allocate arrays contiguously so that they can be used as ring buffer
        void* base_ptr = alloc_gpu_memory(gpu_buffer_depth * len);
        gpu_memory[name].gpu_pointers_to_free.push_back(base_ptr);
        for (uint32_t i = 0; i < gpu_buffer_depth; ++i) {
            // void* ptr = alloc_gpu_memory(len);
            void* ptr = (unsigned char*)base_ptr + i * len;
            INFO("Allocating GPU[{:d}] memory: {:s}, len: {:d}, ptr: {:p}", gpu_id, name, len, ptr);
            gpu_memory[name].len = len;
            gpu_memory[name].view_source = "";
            gpu_memory[name].gpu_pointers.push_back(ptr);
            // gpu_memory[name].gpu_pointers_to_free.push_back(ptr);
            gpu_memory[name].metadata_pointers.push_back(nullptr);
        }
    }
    // The size must match what has already been allocated.
    if (len != gpu_memory[name].len) {
        ERROR("get_gpu_memory_array failed: requested name \"{:s}\" size {:d} index {:d}, but "
              "existing memory is size {:d}",
              name, len, index, gpu_memory[name].len);
    }
    assert(len == gpu_memory[name].len);
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].gpu_pointers.size());

    // Return the requested memory.
    return gpu_memory[name].gpu_pointers[index];
}

void* gpuDeviceInterface::create_gpu_memory_view(const std::string& source_name,
                                                 const size_t source_len,
                                                 const std::string& dest_name, const size_t offset,
                                                 const size_t dest_len) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    // Ensure that the view doesn't already exist
    if (gpu_memory.count(dest_name) > 0)
        throw std::runtime_error(
            fmt::format("Tried to create_gpu_memory_view {:s} that already exists.", dest_name));
    // Get source
    void* source = get_gpu_memory(source_name, source_len);

    // Create dest entry
    INFO("Creating GPU memory view {:s} with length {:d}, view on {:s} + offset {:d}", dest_name,
         dest_len, source_name, offset);
    assert(offset + dest_len < source_len);
    gpu_memory[dest_name].len = dest_len;
    gpu_memory[dest_name].view_source = source_name;
    void* dest = (void*)((unsigned char*)source + offset);
    gpu_memory[dest_name].gpu_pointers.push_back(dest);
    gpu_memory[dest_name].gpu_pointers_to_free.push_back(nullptr);
    return dest;
}

void gpuDeviceInterface::create_gpu_memory_array_view(const std::string& source_name,
                                                      const size_t source_len,
                                                      const std::string& dest_name,
                                                      const size_t offset, const size_t dest_len) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    INFO("Creating GPU memory array view {:s} with length {:d}, view on {:s} + offset {:d}",
         dest_name, dest_len, source_name, offset);
    // Ensure that the view doesn't already exist
    if (gpu_memory.count(dest_name) > 0)
        throw std::runtime_error(fmt::format(
            "Tried to create_gpu_memory_array_view {:s} that already exists.", dest_name));
    assert(offset + dest_len <= source_len);

    for (uint32_t i = 0; i < gpu_buffer_depth; ++i) {
        // Get source
        void* source = get_gpu_memory_array(source_name, i, source_len);
        // Create dest entry
        gpu_memory[dest_name].len = dest_len;
        gpu_memory[dest_name].view_source = source_name;
        gpu_memory[dest_name].gpu_pointers.push_back((unsigned char*)source + offset);
        gpu_memory[dest_name].gpu_pointers_to_free.push_back(nullptr);
        gpu_memory[dest_name].metadata_pointers.push_back(nullptr);
    }
}

void gpuDeviceInterface::create_gpu_memory_ringbuffer(const std::string& source_name,
                                                      const size_t source_len,
                                                      const std::string& dest_name,
                                                      const size_t offset, const size_t dest_len) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    INFO("Creating GPU memory ringbuffer view {:s} with length {:d}, view on {:s} + offset {:d}",
         dest_name, dest_len, source_name, offset);
    // Ensure that the view doesn't already exist
    if (gpu_memory.count(dest_name) > 0)
        throw std::runtime_error(fmt::format(
            "Tried to create_gpu_memory_ringbuffer {:s} that already exists.", dest_name));
    assert(offset + dest_len * gpu_buffer_depth <= source_len);

    // Get/create the full buffer ("source")
    void* source = get_gpu_memory(source_name, source_len);

    // Create "dest" entries (views)
    gpu_memory[dest_name].len = dest_len;
    gpu_memory[dest_name].view_source = source_name;
    for (uint32_t i = 0; i < gpu_buffer_depth; ++i) {
        gpu_memory[dest_name].gpu_pointers.push_back((unsigned char*)source + offset
                                                     + i * dest_len);
        gpu_memory[dest_name].gpu_pointers_to_free.push_back(nullptr);
        gpu_memory[dest_name].metadata_pointers.push_back(nullptr);
    }
}

metadataContainer* gpuDeviceInterface::get_gpu_memory_array_metadata(const std::string& name,
                                                                     const uint32_t index) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    // Memory array must be allocated already
    if (gpu_memory.count(name) == 0) {
        FATAL_ERROR("get_gpu_memory_array_metadata for name \"{:s}\": does not exist yet.", name);
    }
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].metadata_pointers.size());
    // If view, recurse
    if (is_view_of_same_size(name))
        return get_gpu_memory_array_metadata(gpu_memory[name].view_source, index);
    // Return the requested memory (may be NULL)
    metadataContainer* mem = gpu_memory[name].metadata_pointers[index];
    return mem;
}

metadataContainer* gpuDeviceInterface::create_gpu_memory_array_metadata(const std::string& name,
                                                                        const uint32_t index,
                                                                        metadataPool* pool) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    // Memory array must be allocated already
    if (gpu_memory.count(name) == 0) {
        FATAL_ERROR("get_gpu_memory_array_metadata for name \"{:s}\": does not exist yet.", name);
    }
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].metadata_pointers.size());
    // If view, recurse
    if (is_view_of_same_size(name))
        return create_gpu_memory_array_metadata(gpu_memory[name].view_source, index, pool);
    // Make sure the slot isn't occupied.
    assert(gpu_memory[name].metadata_pointers[index] == nullptr);
    // Allocate new metadata obj
    metadataContainer* mc = request_metadata_object(pool);
    assert(mc);
    // Plug it in!
    gpu_memory[name].metadata_pointers[index] = mc;
    return mc;
}

bool gpuDeviceInterface::is_view_of_same_size(const std::string& name) {
    return gpu_memory[name].view_source.size() &&
        (gpu_memory[gpu_memory[name].view_source].metadata_pointers.size() ==
         gpu_memory[name].metadata_pointers.size());
}

void gpuDeviceInterface::claim_gpu_memory_array_metadata(const std::string& name,
                                                         const uint32_t index,
                                                         metadataContainer* mc) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);
    // Memory array must be allocated already
    if (gpu_memory.count(name) == 0) {
        FATAL_ERROR("claim_gpu_memory_array_metadata for name \"{:s}\": does not exist yet.", name);
    }
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].metadata_pointers.size());
    // If view of an array of the same size, recurse
    // (in "ringbuffer" views, the source is a singleton)
    if (is_view_of_same_size(name))
        return claim_gpu_memory_array_metadata(gpu_memory[name].view_source, index, mc);
    // Make sure the slot is empty
    if (gpu_memory[name].metadata_pointers[index]) {
        FATAL_ERROR("claim_gpu_memory_array_metadata for name \"{:s}\"[:d]: slot is not empty.",
                    name, index);
    }
    increment_metadata_ref_count(mc);
    gpu_memory[name].metadata_pointers[index] = mc;
}

void gpuDeviceInterface::release_gpu_memory_array_metadata(const std::string& name,
                                                           const uint32_t index) {
    std::lock_guard<std::recursive_mutex> lock(gpu_memory_mutex);

    // Memory array must be allocated already
    if (gpu_memory.count(name) == 0) {
        FATAL_ERROR("release_gpu_memory_array_metadata for name \"{:s}\": does not exist yet.",
                    name);
    }
    // Make sure we aren't asking for an index past the end of the array.
    assert(index < gpu_memory[name].metadata_pointers.size());
    // If view, recurse
    if (is_view_of_same_size(name))
        return release_gpu_memory_array_metadata(gpu_memory[name].view_source, index);
    metadataContainer* mc = gpu_memory[name].metadata_pointers[index];
    if (mc) {
        decrement_metadata_ref_count(mc);
        gpu_memory[name].metadata_pointers[index] = nullptr;
    }
}
