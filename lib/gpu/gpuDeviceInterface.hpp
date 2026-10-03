#ifndef GPU_DEVICE_INTERFACE_H
#define GPU_DEVICE_INTERFACE_H

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for kotekanLogging
#include "metadata.hpp"       // for metadataObject, metadataPool

#include <map>      // for map
#include <memory>   // for shared_ptr, weak_ptr
#include <mutex>    // for recursive_mutex
#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, int32_t
#include <string>   // for string, basic_string
#include <thread>   // for thread::id
#include <vector>   // for vector

/// Stores a named set of gpu pointer(s) with uniform size
struct gpuMemoryBlock {
    std::vector<void*> gpu_pointers;
    // store the "real" pointers to allow windowed buffer views
    std::vector<void*> gpu_pointers_to_free;
    std::vector<std::shared_ptr<metadataObject>> metadata_pointers;
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
    std::shared_ptr<metadataObject> get_gpu_memory_array_metadata(const std::string& name,
                                                                  const uint32_t index);

    /**
     * @brief Allocates a new metadata object (from the given pool)
     * and attaches it to this GPU array element.
     * @param name  the name of the GPU buffer whose metadata you want to create
     * @param index the GPU buffer array index
     * @param pool  the pool that will be used to create the metadata object
     */
    std::shared_ptr<metadataObject>
    create_gpu_memory_array_metadata(const std::string& name, const uint32_t index,
                                     std::weak_ptr<metadataPool> pool);

    /**
     * @brief Attaches the given metadata to this GPU array element,
     *
     * @param name  the name of the GPU buffer whose metadata you want to create
     * @param index the GPU buffer array index
     * @param mc    the metadata
     */
    void claim_gpu_memory_array_metadata(const std::string& name, const uint32_t index,
                                         std::shared_ptr<metadataObject> mc);


    /**
     * @brief Attribute GPU memory taken on the calling thread to `owner` from now on.
     *
     * Called once from gpuProcess::main_thread(). It is the backstop for a region first taken
     * in execute() by a command that did not register it from its constructor (see
     * register_gpu_memory_name); such a conflict surfaces at the first frame rather than at
     * construction. Registrations last the life of the device object.
     */
    void claim_memory_owner_thread(const std::string& owner);

    /**
     * @brief Record that the current owner will use `name`, before it is allocated.
     *
     * A command that only takes its region in execute() (cudaInputData, cudaOutputData, the ring
     * copies, cudaUpchannelize) calls this from its constructor, so the region is attributed --
     * and a conflict refused -- at construction, where --dry-run can see it.
     */
    void register_gpu_memory_name(const std::string& name);

    /**
     * @brief Declare that the current owner shares `name` with another stage, ordered by
     *        `handshake`.
     *
     * A share is allowed only when BOTH stages declare it with the same handshake. The ring
     * classes (cudaCopyToRingbuffer, cudaCopyFromRingbuffer, cudaCopyNToRingbuffer,
     * NDArrayRingBuffer) pass the host RingBuffer's name: producer and consumer wait on and
     * signal that ring, and the ring is signalled only once the producing stage's frame has
     * completed on the device. Config authors reach the same declaration through
     * `shared_gpu_memory` on the stage (gpuProcess), whose handshake is that key. Must be
     * called with an owner in effect (a stage under construction, or a registered thread).
     */
    void declare_shared_gpu_memory(const std::string& name, const std::string& handshake);

    // Can't do this in the destructor because only the derived classes know
    // how to free their memory. To be moved into distinct objects...
    void cleanup_memory();

    /// This function sets the thread specific variables needed for the GPU API
    /// For example CUDA requires the GPU ID be set per thread
    virtual void set_thread_device() {};

    /// Returns the GPU ID handled by this device object
    int get_gpu_id() {
        return gpu_id;
    }

protected:
    virtual void* alloc_gpu_memory(size_t len) = 0;
    virtual void free_gpu_memory(void*) = 0;

    // This is used internally - when a GPU array is actually a view on another array,
    // then we forward metadata requests, but only if the view is the same size as the
    // original -- so that array-size metadata are still correct.
    bool is_view_of_same_size(const std::string& name);

    // Extra data
    kotekan::Config& config;
    std::string unique_name;

    // Config variables
    int gpu_id;

private:
    /**
     * @brief Attribute GPU memory taken while a stage is being CONSTRUCTED to that stage.
     *
     * Stages are built one at a time on the main thread (StageFactory::build_stages), so one
     * scope per device suffices; gpuProcess::init() holds a gpuMemoryOwnerScope around its
     * command constructors, which is where most named regions are first taken (NDArrayBuffer
     * and NDArrayRingBuffer fetch theirs from member initialisers). Nesting is refused as a bug.
     */
    void begin_memory_owner(const std::string& owner);
    void end_memory_owner();
    friend class gpuMemoryOwnerScope;

    /// Owner of regions taken on each registered enqueuing thread; see claim_memory_owner_thread.
    std::map<std::thread::id, std::string> _thread_owner;

    /// Owner of regions taken while a stage is being constructed; empty outside a scope.
    std::string _constructing_owner;

    /// First owner of each named region.
    std::map<std::string, std::string> _memory_owner;

    /// Declared shares: name -> (declaring stage -> handshake). See declare_shared_gpu_memory.
    std::map<std::string, std::map<std::string, std::string>> _shared_declared;

    /// The stage the calling thread's claims belong to, or empty when there is none.
    const std::string& current_memory_owner() const;

    /// Refuses, by name, a region a second stage takes without a matching declared share.
    void check_memory_claim(const std::string& name);

    std::map<std::string, gpuMemoryBlock> gpu_memory;

    // Mutex to protect gpu_memory variable
    std::recursive_mutex gpu_memory_mutex;
};

/// Holds a construction-owner scope on a device for the lifetime of the object; see
/// gpuDeviceInterface::begin_memory_owner.
class gpuMemoryOwnerScope {
public:
    gpuMemoryOwnerScope(gpuDeviceInterface& device, const std::string& owner) : dev(device) {
        dev.begin_memory_owner(owner);
    }
    ~gpuMemoryOwnerScope() {
        dev.end_memory_owner();
    }
    gpuMemoryOwnerScope(const gpuMemoryOwnerScope&) = delete;
    gpuMemoryOwnerScope& operator=(const gpuMemoryOwnerScope&) = delete;

private:
    gpuDeviceInterface& dev;
};

#endif // GPU_DEVICE_INTERFACE_H
