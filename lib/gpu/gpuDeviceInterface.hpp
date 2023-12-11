#ifndef GPU_DEVICE_INTERFACE_H
#define GPU_DEVICE_INTERFACE_H

#include "Config.hpp"
#include "kotekanLogging.hpp" // for kotekanLogging
#include "metadata.h"

#include <map> // for map
#include <mutex>
#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string
#include <vector>   // for vector

/// Stores a named set of gpu pointer(s) with uniform size
struct gpuMemoryBlock {
    std::vector<void*> gpu_pointers;
    // store the "real" pointers to allow windowed buffer views
    std::vector<void*> gpu_pointers_to_free;
    std::vector<metadataContainer*> metadata_pointers;
    size_t len;
    // if this is a view, the target of that view; used only for metadata
    std::string view_source;
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
    gpuDeviceInterface(kotekan::Config& config, const std::string& unique_name, int32_t gpu_id);

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
    void* get_gpu_memory_array(const std::string& name, const uint32_t index,
                               const uint32_t buffer_depth, const size_t len);

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
                                      const size_t view_len, const uint32_t buffer_depth);

    /**
     * @brief Creates a chunk of GPU memory that is a view on another GPU
     * memory chunk.  (The "source" array need not exist, it will be
     * created with a call to *get_gpu_memory* if it does not exist yet.).
     * Returns the new memory view pointer.
     */
    void* create_gpu_memory_view(const std::string& source_name, const size_t source_len,
                                 const std::string& view_name, const size_t view_offset,
                                 const size_t view_len);

    /**
     * @brief Creates a large GPU memory chunk, and then multiple
     * views into that memory chunk that look like a GPU memory array.
     * This method should be called once during setup.  After this has
     * been called, *get_gpu_memory* for the "source" name will return
     * the full memory chunk, and *get_gpu_memory_array* for the
     * "view" name will return a view into the full memory chunk,
     * where adjacent array indices are contiguous.
     *
     * The "source" singleton and "dest" array each have their own
     * metadata objects.
     *
     * @param source_name like the "name" of get_gpu_memory, the
     *   name of the "real" GPU memory array.
     * @param source_len  the size in bytes of the "real" GPU memory array.
     * @param view_name   the name of the view onto the "real" GPU memory array.
     * @param view_offset the offset in bytes of the views.
     * @param view_len the length in bytes of the views.  *view_offset*
     *   + *view_len* * gpu_buffer_depth must be <= *source_len*.
     */
    void create_gpu_memory_ringbuffer(const std::string& source_name, const size_t source_len,
                                      const std::string& view_name, const size_t view_offset,
                                      const size_t view_len, const uint32_t buffer_depth);

    /**
     * @brief Fetches the metadata (if any) attached to the given GPU
     * memory array element.  Return NULL if no metadata.
     * @param name  the name of the GPU buffer whose metadata you want
     * @param index the GPU buffer array index
     */
    metadataContainer* get_gpu_memory_array_metadata(const std::string& name, const uint32_t index);

    /**
     * @brief Allocates a new metadata object (from the given pool)
     * and attaches it to this GPU array element.
     * @param name  the name of the GPU buffer whose metadata you want to create
     * @param index the GPU buffer array index
     * @param pool  the pool that will be used to create the metadata object
     */
    metadataContainer* create_gpu_memory_array_metadata(const std::string& name,
                                                        const uint32_t index, metadataPool* pool);

    /**
     * @brief Attaches the given metadata to this GPU array element,
     * incrementing the reference count.  This should be accompanied
     * by a *release_gpu_memory_array_metadata* call to release the
     * reference count.
     * @param name  the name of the GPU buffer whose metadata you want to create
     * @param index the GPU buffer array index
     * @param mc    the metadata container whose ref count will get increased
     */
    void claim_gpu_memory_array_metadata(const std::string& name, const uint32_t index,
                                         metadataContainer* mc);

    /**
     * @brief Releases a claim on a metadata object (including a newly
     * created metadata object) attached to the given GPU array
     * element.  Does nothing if no metadata object has been attached
     * to this GPU array element.
     */
    void release_gpu_memory_array_metadata(const std::string& name, const uint32_t index);

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

protected:
    virtual void* alloc_gpu_memory(size_t len) = 0;
    virtual void free_gpu_memory(void*) = 0;

    bool is_view_of_same_size(const std::string& name);

    // Extra data
    kotekan::Config& config;
    std::string unique_name;

    // Config variables
    int gpu_id;

private:
    std::map<std::string, gpuMemoryBlock> gpu_memory;

    // Mutex to protect gpu_memory variable
    std::recursive_mutex gpu_memory_mutex;
};

#endif // GPU_DEVICE_INTERFACE_H
