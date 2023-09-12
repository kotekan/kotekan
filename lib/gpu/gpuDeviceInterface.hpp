#ifndef GPU_DEVICE_INTERFACE_H
#define GPU_DEVICE_INTERFACE_H

#include "Config.hpp"
#include "kotekanLogging.hpp" // for kotekanLogging

#include <map>      // for map
#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string
#include <vector>   // for vector

/// Stores a named set of gpu pointer(s) with uniform size
struct gpuMemoryBlock {
    std::vector<void*> gpu_pointers;
    // store the "real" pointers to allow windowed buffer views
    std::vector<void*> gpu_pointers_to_free;
    size_t len;
};

/**
 * @class gpuDeviceInterface
 * @brief Base class for interacting with GPU devices.
 *        Primarily deals with memory allocation in GPU subsystems.
 *
 * @author Keith Vanderlinde
 */
class gpuDeviceInterface : public kotekan::kotekanLogging {
public:
    /// Constructor
    gpuDeviceInterface(kotekan::Config& config, const std::string& unique_name, int32_t gpu_id,
                       int gpu_buffer_depth);
    /// Destructor
    virtual ~gpuDeviceInterface();

    /**
     * @brief Get one of the gpu memory pointers with the given name and size = len at the given
     * index The size of the set is equal to _gpu_buffer_depth, so index < _gpu_buffer_depth If a
     * region with this name exists then it will just return an existing pointer at the give index,
     * if the region doesn't exist, then it creates it with gpu_buffer_depth pointers of size len
     * NOTE: if accessing an existing named region then len must match the existing
     * length or the system will throw an assert.
     */
    void* get_gpu_memory_array(const std::string& name, const uint32_t index, const size_t len);

    /**
     * @brief Same as get_gpu_memory_array but gets just one gpu memory buffer
     * This can be used when internal memory is needed.
     * i.e. memory used for lookup tables that are the same between runs
     * or temporary buffers between kernels.
     * Should NOT be used for any memory that's copied between GPU and HOST memory.
     */
    void* get_gpu_memory(const std::string& name, const size_t len);

    /**
     * @brief Creates a GPU memory array that is a view on another GPU
     * memory array.  (The "source" array need not exist, it will be
     * created if it does not exist yet.)  That is, the "view" does
     * not allocate new GPU memory, but rather exposes a sub-array as
     * though it was an independent array.  This allows, for example,
     * one GPU stage to write into a (view on a) memory region that
     * has padding regions on the front or back, or one stage to read
     * a subset of the output produced by a previous stage, in a
     * transparent manner.
     * This method should be called once during setup.
     * After this has been called, *get_gpu_memory_array* calls for the "source_name"
     * or "view_name" will return the real or view memory pointers.
     *
     * @param source_name like the "name" of get_gpu_memory_array, the
     *   name of the "real" GPU memory array.
     * @param source_len  the size in bytes of the "real" GPU memory array.
     * @param view_name   the name of the view onto the "real" GPU memory array.
     * @param view_offset the offset in bytes of the view.
     * @param view_len the length in bytes of the view.  *view_offset*
     *   + *view_len* must be <= *source_len*.
     */
    void create_gpu_memory_array_view(const std::string& source_name, const size_t source_len,
                                      const std::string& view_name, const size_t view_offset,
                                      const size_t view_len);

    /**
     * @brief Creates a chunk of GPU memory that is a view on another GPU
     * memory chunk.  (The "source" array need not exist, it will be
     * created with a call to *get_gpu_memory* if it does not exist yet.).
     * Returns the new memory view pointer.
     */
    void* create_gpu_memory_view(const std::string& source_name, const size_t source_len,
                                 const std::string& view_name, const size_t view_offset,
                                 const size_t view_len);

    // Can't do this in the destructor because only the derived classes know
    // how to free their memory. To be moved into distinct objects...
    void cleanup_memory();

    /// This function sets the thread specific variables needed for the GPU API
    /// For example CUDA requires the GPU ID be set per thread
    virtual void set_thread_device(){};

    /// Returns the GPU ID handled by this device object
    int get_gpu_id() {
        return gpu_id;
    }

    /// Returns the gpu buffer depth
    int get_gpu_buffer_depth() {
        return gpu_buffer_depth;
    }

protected:
    virtual void* alloc_gpu_memory(size_t len) = 0;
    virtual void free_gpu_memory(void*) = 0;

    // Extra data
    kotekan::Config& config;
    std::string unique_name;

    // Config variables
    int gpu_id;
    uint32_t gpu_buffer_depth;

private:
    std::map<std::string, gpuMemoryBlock> gpu_memory;
};

#endif // GPU_DEVICE_INTERFACE_H
