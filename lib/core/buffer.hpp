/**
 * @file
 * @brief The core kotekan buffer object for data transfer between stages
 *  - buffer
 *  - StageInfo
 *  - create_buffer
 *  - delete_buffer
 *  - zero_frames
 *  - register_consumer
 *  - register_producer
 *  - mark_frame_full
 *  - mark_frame_empty
 *  - wait_for_empty_frame
 *  - wait_for_full_frame
 *  - is_frame_empty
 *  - get_num_full_frames
 *  - print_buffer_status
 *  - allocate_new_metadata_object
 *  - get_metadata
 *  - get_metadata_container
 *  - pass_metadata
 *  - send_shutdown_signal
 */

#ifndef BUFFER
#define BUFFER

#include "kotekanLogging.hpp"
#include "metadata.h" // for metadataPool

#include <pthread.h> // for pthread_cond_t, pthread_mutex_t
#include <stdbool.h> // for bool
#include <stdint.h>  // for uint8_t
#include <time.h>    // for size_t, timespec

#include <map>
#include <vector>
#include <string>

#ifdef MAC_OSX
#include "osxBindCPU.hpp"

#include <immintrin.h>
#endif

/// The system page size, this might become more dynamic someday
#define PAGESIZE_MEM 4096

/**
 * @struct StageInfo
 * @brief Internal structure for tracking consumer and producer names.
 */
class StageInfo {
public:
    // Non-copyable signatures
    StageInfo(const std::string& _name, int _num_frames) :
        name(_name),
        num_frames(_num_frames),
        last_frame_acquired(-1),
        last_frame_released(-1),
        is_done(_num_frames) {
    }
    // moves allowed
    //StageInfo(StageInfo&&) = default;
    //StageInfo& operator=(StageInfo&&) = default;
    StageInfo(StageInfo&&) = delete;
    StageInfo& operator=(StageInfo&&) = delete;
    // you shall not copy
    StageInfo(const StageInfo&) = delete;
    StageInfo& operator=(const StageInfo&) = delete;
    
    /// Set to 1 if the stage is active
    //int in_use;

    /// The name of the stage (consumer or producer)
    const std::string name;

    int num_frames;

    /// Last frame acquired with a call to wait_for_*
    int last_frame_acquired;

    /// Last frame to be released with a call to mark_frame_*
    int last_frame_released;

    std::vector<bool> is_done;
};

class GenericBuffer  : public kotekan::kotekanLogging {
public:
    GenericBuffer(const std::string&,
                  const std::string&,
                  metadataPool* pool,
                  int num_frames);
    virtual ~GenericBuffer();
    virtual void print_full_status() {};

    virtual bool is_basic() = 0;

    /**
     * @brief Tells the buffers to stop returning full/empty frames to consumers/producers
     *
     * This function should only be called by the framework, and not by stages.
     * Once called it will cause all @c wait_for_empty_frame() and @c wait_for_full_frame()
     * calls to wake up and return NULL; or NULL on the next time they are called.
     */
    void send_shutdown_signal();

    /**
     * @brief Register a consumer with a given name.
     *
     * In order to use a buffer a consumer must first register its name so that
     * the buffer object can track which consumers have signed off on each frame.
     *
     * @param[in] name The name of the consumer.
     */
    void register_consumer(const std::string& name);

    /**
     * @brief Removes the consumer with the given name
     *
     * In some cases it may make sense to stop being a consumer of a given
     * buffer while the pipeline is running.  However this is likely an edge
     * case for most pipelines.  In general it is not expected for stages
     * to unregister when they close.
     *
     * @param name The name of the consumer to unregister
     */
    void unregister_consumer(const std::string& name);

    /**
     * @brief Register a producer with a given name.
     *
     * In order to use a buffer a producer must first register its name so that
     * the buffer object can track which producers have signed off on each frame.
     *
     * @param[in] name The name of the producer.
     */
    void register_producer(const std::string& name);

    /**
     * @brief Get the number of consumers on this buffer
     *
     * @return int The number of consumers on the buffer
     */
    int get_num_consumers();

    /**
     * @brief Get the number of producers for this buffer
     *
     * @return int The number of producers on this buffer
     */
    int get_num_producers();

    /// The main lock for frame state management
    pthread_mutex_t lock;

    /// The condition variable for calls to @c wait_for_full_buffer
    pthread_cond_t full_cond;

    /// The condition variable for calls to @c wait_for_empty_buffer
    pthread_cond_t empty_cond;

    /// The number of frames kept by this object
    int num_frames;

    /**
     * @brief Shutdown variable
     * Set to 1 when the system should stop returning
     * new frames for producers and consumers.
     */
    int shutdown_signal;

    /// The list of consumer names registered to this buffer
    std::map<std::string, StageInfo> consumers;

    /// The list of producer names registered to this buffer
    std::map<std::string, StageInfo> producers;

    /// The name of the buffer for use in debug messages.
    std::string buffer_name;

    /// The type of the buffer for use in writing data.
    std::string buffer_type;

    /// The pool of info objects
    metadataPool* metadata_pool;

protected:
    //virtual void registered_producer(StageInfo&) {}
    
};

class RingBuffer : public GenericBuffer {
public:
    //size_t, 
    RingBuffer(metadataPool*, const std::string&,
               const std::string&); //, int, bool, bool, bool);
    ~RingBuffer() override {}
    bool is_basic() override { return false; }
};

/**
 * @struct Buffer
 * @brief Kotekan's core multi-producer, multi-consumer ring buffer with metadata
 *
 * This class is the central method for passing data between kotekan stages
 * in a pipeline.
 *
 * It provides a fixed size RING buffer which can have multiple producers and
 * consumers attached to it.   The idea is that individual stages do not need
 * to worry about how to manage data transfer between them, it is taken care of by
 * this class.  All the public functions here are thread safe, and if used correctly
 * tested to be deadlock free.
 *
 * The terminology here is a "frame" is a block of memory of size @c frame_size
 * and the buffer consists of a ring with @c num_frames independent frames.
 * When a frame is available for producers to add data to, it is considered "empty"
 * When a frame is ready to be read by consumers is it considered "full"
 * Therefore a producer asks for an "empty" frame, and marks it as "full" when done.
 * Likewise a consumer asks for a "full" frame and marks it as "empty" when done.
 *
 * There can be more than one producer or consumer attached to each buffer, but
 * each one must register with the buffer separately.
 *
 * Consumers must only read data from frames and not write anything back to them.
 * More than one producer can write to a given frame in a multi-producer setup,
 * but in that case they must coordinate their address space to not overwrite
 * each others' values.  Because of this, multi-producers are somewhat rare.
 * Producers also generally shouldn't read from frames, although there is
 * nothing wrong with doing so, it just normally doesn't make sense to do so.
 *
 * Unless the function @c zero_frames() is called on the buffer object, the
 * default behaviour is not to zero the memory of the frames between uses.
 * Therefore it is normally up to the producer(s) to ensure all memory
 * values are either given new data, or zeroed.
 *
 * In the config file a buffer is created with a <tt>kotekan_buffer: standard</tt>
 * named block.   The buffer name becomes the path name of that config block.
 *
 * Note that if no consumer is registered for on a buffer, then it will drop
 * the frames and log an INFO statement to notify the user that the data
 * is being dropped.
 *
 * @conf frame_size The size of the individual ring frames in bytes
 * @conf num_frames The buffer depth of size of the ring
 * @conf metadata_pool The name of the metadata pool to associate with the buffer
 * @conf numa_node The NUMA domain to mbind the memory into.  Default: 1
 * @conf use_hugepages Allocate 2MB huge pages for the frames. Default: false
 * @conf mlock_frames Lock the frame pages with mlock Default: true
 *
 * See metadata.h for more information on metadata pools
 *
 * @author Andre Renard
 */
class Buffer : public GenericBuffer {
public:
    Buffer(int num_frames, size_t len, metadataPool* pool, const std::string& buffer_name,
           const std::string& buffer_type, int numa_node, bool use_hugepages, bool mlock_frames,
           bool zero_new_frames);
    ~Buffer() override;

    bool is_basic() override { return true; }
    
    /**
     * @brief Prints a summary the frames and state of the producers and consumers.
     *
     * @param buf The buffer object
     */
    void print_full_status() override;

    //protected:
    /// The size of each frame in bytes.
    size_t frame_size;

    /**
     * @brief The padded frame size.
     * Each frame is padded out to a page aligned size,
     * your operation wants to do things paged aligned
     * you can use this size instead, but data shouldn't
     * be placed past the end of frame_size.  This is just for padding.
     */
    size_t aligned_frame_size;

    /**
     * @brief Array of producers which are done (marked frame as full).
     * Format is [ID][producer]
     * zero means not done, 1 means done (marked as full)
     */
    //int** producers_done;
    //std::map<std::string, std::vector<bool> > producers_done;

    /**
     * @brief Array of consumers which are done (marked frame as empty).
     * Format is [ID][consumer]
     * zero means not done, 1 means done (marked as empty)
     */
    //int** consumers_done;

    /// Flag set to indicate if the frames should be zeroed between uses
    int zero_frames;

    /// The array of frames (the actual data we are carrying)
    uint8_t** frames;

    /**
     * @brief Flag variables to say which frames are full
     * A 0 at index I means the frame at index I is not full, one means it is full.
     */
    int* is_full;

    /// The last time a frame was marked as full (used for arrival rate)
    double last_arrival_time;

    /// Array of buffer info objects, for tracking information about each buffer.
    metadataContainer** metadata;

    /// This buffer use huge pages for its frames if the following is true
    bool use_hugepages;

    /// The buffer has page locked memory frames
    bool mlock_frames;

    /// The NUMA node the frames are allocated in
    int numa_node;

    /// Ring buffers only, SINGLE CONSUMER:
    size_t ring_buffer_size;
    size_t ring_buffer_elements;
    size_t ring_buffer_elements_claimed;
    size_t ring_buffer_read_size;
    size_t ring_buffer_write_cursor;
    size_t ring_buffer_read_cursor;
    size_t ring_buffer_full_frame;
};

/**
 * @brief Creates a buffer object.
 *
 * Used to create a buffer object, normally invoked by the buffer factory
 * as a part of the pipeline generation from the config file, not intended
 * to be called directly.
 *
 * @param[in] num_frames The number of frames to create in the buffer ring.
 * @param[in] frame_size The length of each frame in bytes.
 * @param[in] pool The metadataPool, which may be shared between more than one buffer.
 * @param[in] buffer_name The unique name of this buffer.
 * @param[in] buffer_type The type of data this buffer contains.
 * @param[in] numa_node The CPU NUMA memory region to allocate memory in.+
 * @param[in] use_huge_pages Map huge pages with mmap
 * @param[in] mlock_frames If set, mlock the pages of the frame memory
 * @param[in] zero_new_frames In theory some memory allocators don't zero new allocations
 *                            so by default we zero new frames on startup, but this is expensive
 *                            and can be disabled by setting this to false.
 * @returns A buffer object.
 */
Buffer* create_buffer(int num_frames, size_t frame_size, metadataPool* pool,
                      const std::string& buffer_name, const std::string& buffer_type, int numa_node,
                      bool use_huge_pages, bool mlock_frames, bool zero_new_frames);

/**
 * @brief Deletes a buffer object and frees all frame memory
 *
 * @param[in] buf The buffer to delete.
 */
void delete_buffer(Buffer* buf);

/**
 * @brief Zero all frames after all consumers have marked them as empty
 *
 * @param[in] buf The buffer object which will be set to automatically zero all frames
 */
void zero_frames(Buffer* buf);

/**
 * @brief Marks a buffer frame as full.
 *
 * This function is used by a producer to sign off that it will no longer write
 * data to this frame.
 *
 * @param[in] buf The buffer containing the frame to mark as full
 * @param[in] producer_name The name of the producer registered with @c register_producer()
 * @param[in] frame_id The frame ID to be marked as full
 */
void mark_frame_full(Buffer* buf, const char* producer_name, const int frame_id);

/**
 * @brief Marks a buffer frame as empty
 *
 * Used by a consumer to sign off that it will no longer read data from this frame.
 *
 * @param[in] buf The buffer containing the frame to mark as empty
 * @param[in] consumer_name The name of the consumer registered with @c register_consumer()
 * @param[in] frame_id The frame ID to be marked as empty
 */
void mark_frame_empty(Buffer* buf, const char* consumer_name, const int frame_id);

/**
 * @brief Blocks until the frame requested by frame_id is empty.
 *
 * This blocking function will return only when the frame_id request is marked
 * as empty internally, or the function @c send_shutdown_signal() is called, which
 * causes the function to return a @c NULL pointer.
 * Generally a stage should exit and cleanup if NULL is returned.
 *
 * @param[in] buf The buffer object
 * @param[in] producer_name The name of the registered producer requesting the frame_id
 * @param[in] frame_id The id of the frame to wait for.
 * @returns A pointer to the frame, or NULL if the buffer is shutting down.
 * @warning After calling this function for a given producer and frame_id it
 *          should not be called again on that frame_id until after
 *          a call to @c mark_frame_full() with that producer and frame_id
 */
uint8_t* wait_for_empty_frame(Buffer* buf, const char* producer_name, const int frame_id);

/// RING BUFFER

void buffer_set_ring_buffer_size(Buffer* buf, size_t sz);

void buffer_set_ring_buffer_read_size(Buffer* buf, size_t sz);

int buffer_wait_for_ring_buffer_writable(Buffer* buf, size_t sz);

void buffer_wrote_to_ring_buffer(Buffer* buf, size_t sz);

uint8_t* buffer_claim_next_full_frame(Buffer* buf, const char* consumer_name, const int frame_id);

/**
 * @brief Blocks until the frame requested by frame_id is full.
 *
 * This blocking function will return only when the frame_id request is marked
 * as full internally, or the function @c send_shutdown_signal() is called, which
 * causes the function to return a @c NULL pointer.
 * Generally a stage should exit and cleanup if NULL is returned.
 *
 * @param[in] buf The buffer object
 * @param[in] consumer_name The name of the registered producer requesting the frame_id
 * @param[in] frame_id The id of the frame to wait for.
 * @returns A pointer to the frame, or NULL if the buffer is shutting down.
 * @warning After calling this function for a given consumer and frame_id it
 *          should not be called again on that frame_id until after
 *          a call to @c mark_frame_empty() with that consumer and frame_id
 */
uint8_t* wait_for_full_frame(Buffer* buf, const char* consumer_name, const int frame_id);


/**
 * @brief Wait for a full frame on the given buffer up to timeout.
 *
 * This function will timeout after `wait` seconds.
 *
 * @param[in] buf Buffer to wait on.
 * @param[in] name Name of the stage.
 * @param[in] ID Frame ID to wait at.
 * @param[in] timeout Exit after this we exceed this *absolute* time.
 *
 * @return Return status:
 *   - `0`: Success! We have a new frame.
 *   - `1`: Failure! We timed out waiting.
 *   - `-1`: Failure! We received the thread exit signal.
 **/
int wait_for_full_frame_timeout(Buffer* buf, const char* name, const int ID,
                                const struct timespec timeout);

/**
 * @brief Checks if the requested buffer is empty.
 *
 * Returns 1 if the buffer is empty, and 0 if the frame is full.
 *
 * @param[in] buf The buffer object
 * @param[in] frame_id The id of the frame to check.
 * @warning This should not be used to gain access to an empty frame, use @c wait_for_empty_frame()
 */
int is_frame_empty(Buffer* buf, const int frame_id);

/**
 * @brief Returns the number of currently full frames.
 *
 * @param[in] buf The buffer object
 * @returns The number of currently full frames in the buffer
 */
int get_num_full_frames(Buffer* buf);

/**
 * @brief Returns the last time a frame was marked as full
 * @param buf The buffer to get the last arrival time for.
 * @return A double (with units: seconds) containing the unix time of the last frame arrival
 */
double get_last_arrival_time(Buffer* buf);

/**
 * @brief Prints a picture of the frames which are currently full.
 *
 * @param[in] buf The buffer object
 */
void print_buffer_status(Buffer* buf);

/**
 * @brief Allocates a new metadata object from the associated pool
 *
 * Needs to be called by the first producer in a chain, or by a producer
 * generating a new type of metadata for the next stage.  If the producer is
 * just passing the metadata down stream from the input buffer use @c pass_metadata()
 *
 * The metadata type is based on the pool type associated with the buffer object
 *
 * @param[in] buf The buffer object
 * @param[in] frame_id The frame ID to assign a metadata object too.
 */
void allocate_new_metadata_object(Buffer* buf, int frame_id);

/**
 * @brief Swaps the provided frame of memory with the internal frame
 *        given by @c frame_id
 *
 * @warning The frame returned will no longer be controlled by this buffer,
 *          and so must be freeded by the system taking it.  Also the frame
 *          given will be used and freed by the buffer, so the providing system
 *          must not attempt to free it.
 * @warning This function should only be used by single producer stages.
 * @warning The extra frame provided to this function must be allocated with
 *          @c buffer_malloc() and the frame returned by this function must be
 *          freed with @c buffer_free()
 * @warning Take care when using this function!
 *
 * @param buf The buffer object to swap with
 * @param frame_id The frame to swap
 * @param external_frame The extra frame to use in place of the existing internal frame.
 * @return The internal frame
 */
uint8_t* swap_external_frame(Buffer* buf, int frame_id, uint8_t* external_frame);

/**
 * @brief Swaps frames between two buffers with identical size for the given frame_ids
 *
 * This function does not swap metadata.  That should be passed with the @c pass_metadata function
 *
 * @warning This function should only be used with a single consumer @c from_buf, and given to a
 *          single producer @c to_buf.
 * @warning The buffer sizes must be identical.
 * @warning Take care with this function!
 *
 * @param from_buf The buffer to take the frame from, and swap with the @c to_buf frame.
 * @param from_frame_id The frame ID to move to the @c to_buf
 * @param to_buf The buffer to take the frame from @c from_buf
 * @param to_frame_id The frame to replace with the frame from @c from_buf
 */
void swap_frames(Buffer* from_buf, int from_frame_id, Buffer* to_buf, int to_frame_id);

/**
 * @brief Allocates a frame with the required malloc method
 *
 * @param len The size of the frame to allocate in bytes.
 * @param numa_node The CPU NUMA region to allocate the memory in.
 * @param use_huge_pages Use mmap to allocate huge pages for frames
 * @param memlock_frames Use mlock to lock frame pages
 * @param zero_new_frames If true, new frames are zeroed with memset
 * @return A pointer to the new memory, or @c NULL if allocation failed.
 */
uint8_t* buffer_malloc(size_t len, int numa_node, bool use_huge_pages, bool memlock_frames,
                       bool zero_new_frames);

/**
 * @brief Deallocate a frame of memory with the required free method.
 *
 * @param frame_pointer The pointer to the memory to free.
 * @param size The size of the memory space to free (needed for NUMA)
 * @param use_huge_pages Toggles the type of "free" call used, must match @c buffer_malloc type
 */
void buffer_free(uint8_t* frame_pointer, size_t size, bool use_huge_pages);

/**
 * @brief Gets the raw metadata block for the given frame
 *
 * Returns a raw <tt>void *</tt> pointer which can then be cast as the
 * the metadata type associated with the buffer.
 *
 * @warning Only call this function for a @c frame_id for which you have
 * access via a call to @c wait_for_full_frame() and use this metadata before
 * calling @p mark_frame_empty(), because it could be dereferenced and returned
 * to the metadata pool after that call.
 * If you are adding a new metadata object, please *also* call
 * @c allocate_new_metadata_object() before asking for the metadata object.
 *
 * @param[in] buf The buffer object
 * @param[in] frame_id The frame to return the metadata for.
 * @returns A pointer to the metadata object (needs to be cast)
 */
void* get_metadata(Buffer* buf, int frame_id);

/**
 * @brief Returns the container for the metadata.
 *
 * This works exactly the same way as @p get_metadata() but returns a
 * @c metadataContainer which holds the reference count, locks, etc.
 *
 * @warning Only call this function for a @c frame_id for which you have
 * access via a call to @c wait_for_full_frame() and use this metadata before
 * calling @p mark_frame_empty(), because it could be dereferenced and returned
 * to the metadata pool after that call.
 * If you are adding a new metadata object, please *also* call
 * @c allocate_new_metadata_object() before asking for the metadata object.
 *
 * @param[in] buf The buffer object
 * @param[in] frame_id The frame to return the metadata for.
 * @returns A pointer to the metadata_container
 */
metadataContainer* get_metadata_container(Buffer* buf, int frame_id);

/**
 * @brief Transfers metadata from one buffer to another for a given frame.
 *
 * This function is used by threads which are both consumers and producers
 * and need to pass the metadata down the pipeline.
 *
 * It should be called only after acquiring both a full frame to read from
 * and an empty frame to copy into. Using @c wait_for_full_frame and @c wait_for_empty_frame
 *
 * Note it doesn't actually copy the metadata, instead it just copies the pointer
 * and uses reference counting to track which buffers (and frames) have registered
 * access to this metadata object.  The stage releases the metadata implicitly
 * when the @c mark_frame_empty() function is called, which decrements
 * the reference counter. Once it reaches zero, the the metadata is returned to the
 * pool.
 *
 * @param[in] from_buf The buffer to copy the metadata from
 * @param[in] from_frame_id The frame ID to copy the metadata from
 * @param[in] to_buf The buffer to copy the metadata into
 * @param[in] to_frame_id The frame ID in the @c to_buf to copy the metadata into
 */
void pass_metadata(Buffer* from_buf, int from_frame_id, Buffer* to_buf, int to_frame_id);

/**
 * @brief Makes a fully deep copy of the metadata from one object to another
 *
 * Unlike pass_metadata this doesn't remove the metadata from the @c from_buf
 * and requires that the @c to_buf has a metadata object to be copied into.
 *
 * @param[in] from_buf The buffer to copy the metadata from
 * @param[in] from_frame_id The frame ID to copy the metadata from
 * @param[in] to_buf The buffer to copy the metadata into
 * @param[in] to_frame_id The frame ID in the @c to_buf to copy the metadata into
 */
void copy_metadata(Buffer* from_buf, int from_frame_id, Buffer* to_buf, int to_frame_id);

/**
 * @brief Swaps a frame or performs a deep copy depending on the number of consumers on the
 *        source buffer.
 *
 * Like @c swap_frames(), but doesn't fail if there is more than one consumer on the source buffer.
 * Does not pass or copy metadata.
 *
 * @param[in] src_buf The source buffer
 * @param[in] src_frame_id The source frame ID
 * @param[in] dest_buf The destination buffer
 * @param[in] dest_frame_id The destination frame ID
 */
void safe_swap_frame(Buffer* src_buf, int src_frame_id, Buffer* dest_buf, int dest_frame_id);

#endif
