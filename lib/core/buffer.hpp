/**
 * @file
 * @brief The core kotekan buffer objects for data transfer between stages
 *  - GenericBuffer
 *  - Buffer
 *  - StageInfo (producers/consumers)
 */
#ifndef KOTEKAN_BUFFER_HPP
#define KOTEKAN_BUFFER_HPP

#include "kotekanLogging.hpp"
#include "metadata.hpp" // for metadataPool

#include "json.hpp" // for basic_json<>::object_t, basic_json<>::value_type, json

#include <condition_variable>
#include <map>
#include <mutex>
#include <stdbool.h> // for bool
#include <stdint.h>  // for uint8_t
#include <string>
#include <time.h> // for size_t, timespec
#include <vector>

#ifdef MAC_OSX
#include "osxBindCPU.hpp"

#include <immintrin.h>
#endif

/// The system page size, this might become more dynamic someday
#define PAGESIZE_MEM 4096

/**
 * @struct StageInfo
 * @brief Internal structure for tracking consumer and producer names and status.
 */
class StageInfo {
public:
    StageInfo(const std::string& _name, int _num_frames) :
        name(_name), num_frames(_num_frames), last_frame_acquired(-1), last_frame_released(-1),
        is_done(_num_frames) {}
    // No copies or moves allowed
    StageInfo(StageInfo&&) = delete;
    StageInfo& operator=(StageInfo&&) = delete;
    StageInfo(const StageInfo&) = delete;
    StageInfo& operator=(const StageInfo&) = delete;

    /// The name of the stage (consumer or producer)
    const std::string name;

    /// The number of frames being processed by this object
    int num_frames;

    /// Last frame acquired with a call to wait_for_*
    int last_frame_acquired;

    /// Last frame to be released with a call to mark_frame_*
    int last_frame_released;

    // Is this producer/consumer done with each frame?
    std::vector<bool> is_done;
};

/**
 * @brief Top-level generic (abstract) buffer type.  This has the concept
 * of multiple producers and consumers, metadata, and signals to be
 * used between producers and consumers.
 *
 * Buffers are the central method for passing data between kotekan stages
 * in a pipeline.
 *
 * All the public functions here are thread safe, and if used correctly
 * tested to be deadlock free.
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
 * Note that if no consumer is registered for on a buffer, then
 * kotekan will drop the frames and log an INFO statement to notify
 * the user that the data is being dropped.
 */
class GenericBuffer : public kotekan::kotekanLogging {
public:
    /**
     * @brief Common-core buffer class.
     *
     * @param buffer_name Unique name for this buffer based on location in config file
     * @param buffer_type Type name, eg "standard", "vis", "hfb", "ring"
     * @param num_frames The buffer depth (for subclasses that have that concept)
     * @param metadata_pool The name of the metadata pool to associate with the buffer
     */
    GenericBuffer(const std::string& buffer_name, const std::string& buffer_type,
                  std::shared_ptr<metadataPool> pool, int num_frames);

    virtual ~GenericBuffer();

    /**
     * @brief Prints a summary the frames and state of the producers and consumers.
     */
    virtual void print_full_status() {}

    /**
     * @brief Prints a picture of this buffer.
     */
    virtual void print_buffer_status() {}

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
    virtual void register_consumer(const std::string& name);

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
    virtual void unregister_consumer(const std::string& name);

    /**
     * @brief Register a producer with a given name.
     *
     * In order to use a buffer a producer must first register its name so that
     * the buffer object can track which producers have signed off on each frame.
     *
     * @param[in] name The name of the producer.
     */
    virtual void register_producer(const std::string& name);

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

    /**
     * @brief Allocates a new metadata object from the associated pool
     *
     * Needs to be called by the first producer in a chain, or by a producer
     * generating a new type of metadata for the next stage.  If the producer is
     * just passing the metadata down stream from the input buffer use @c pass_metadata()
     *
     * The metadata type is based on the pool type associated with the buffer object
     *
     * @param[in] frame_id The frame ID to assign a metadata object
     * to. (Zero for subclasses that have no such concept)
     */
    void allocate_new_metadata_object(int frame_id);

    /**
     * @brief Sets the metadata object for the given frame index.
     *
     * @return @c true if successful, @c false if metadata was already set.
     */
    bool set_metadata(int frame_id, std::shared_ptr<metadataObject> meta);

    /**
     * @brief Gets the metadata block for the given frame
     *
     * Returns a shared_ptr to metadataObject which can then be cast as the
     * the metadata type associated with the buffer.
     *
     * @warning Only call this function for a @c frame_id for which you have
     * access via a call to @c wait_for_full_frame() and use this metadata before
     * calling @p mark_frame_empty(), because it could be dereferenced and returned
     * to the metadata pool after that call.
     * If you are adding a new metadata object, please *also* call
     * @c allocate_new_metadata_object() before asking for the metadata object.
     *
     * @param[in] frame_id The frame to return the metadata for.
     * @returns A pointer to the metadata object
     */
    std::shared_ptr<metadataObject> get_metadata(int frame_id);

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
     * @param[in] from_frame_id The frame ID to copy the metadata from
     * @param[in] to_buf The buffer to copy the metadata into
     * @param[in] to_frame_id The frame ID in the @c to_buf to copy the metadata into
     */
    void pass_metadata(int from_ID, GenericBuffer* to_buf, int to_ID);

    /**
     * @brief Makes a fully deep copy of the metadata from one object to another
     *
     * Unlike pass_metadata this doesn't remove the metadata from the @c from_buf
     * and requires that the @c to_buf has a metadata object to be copied into.
     *
     * @param[in] from_frame_id The frame ID to copy the metadata from
     * @param[in] to_buf The buffer to copy the metadata into
     * @param[in] to_frame_id The frame ID in the @c to_buf to copy the metadata into
     */
    void copy_metadata(int from_frame_id, GenericBuffer* to_buf, int to_frame_id);

    /**
     * @brief Populates a JSON description of this buffer.
     *
     * @param buf_json The JSON object to populate.
     */
    virtual void json_description(nlohmann::json& buf_json);

    /**
     * @brief Returns a text description of this buffer for use in the automatic Dot pipeline
     * graphs.
     */
    virtual std::string get_dot_node_label();

    /// The number of frames kept by this object
    int num_frames;

    /// Should we shut down (stop returning new frames)?
    bool shutdown_signal;

    /// The list of consumer names registered to this buffer
    std::map<std::string, StageInfo> consumers;

    /// The list of producer names registered to this buffer
    std::map<std::string, StageInfo> producers;

    /// The name of the buffer for use in debug messages.
    std::string buffer_name;

    /// The type of the buffer for use in writing data.
    std::string buffer_type;

    /// The pool of info objects
    std::shared_ptr<metadataPool> metadata_pool;

    /// Array of buffer info objects, for tracking information about each buffer.
    std::vector<std::shared_ptr<metadataObject>> metadata;

protected:
    /// The main lock for frame state management
    std::recursive_mutex mutex;

    /// The condition variable for calls to @c wait_for_full_buffer
    std::condition_variable_any full_cond;

    /// The condition variable for calls to @c wait_for_empty_buffer
    std::condition_variable_any empty_cond;

    void private_copy_metadata(int dest_frame_id, GenericBuffer* src, int src_frame_id);
};

/**
 * @brief Kotekan's core multi-producer, multi-consumer ring buffer with metadata.
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
    /**
     * @brief Construct a Buffer
     *
     * @param num_frames - number of "frames" in this ring buffer
     * @param len - length in bytes of each frame
     * @param metadata_pool The name of the metadata pool to associate with the buffer
     * @param buffer_name: unique name for this buffer, from the config file declaration
     * @param buffer_type: "standard", "vis", "hfb"
     * @param numa_node The NUMA domain to mbind the memory into
     * @param use_hugepages Allocate 2MB huge pages for the frames
     * @param mlock_frames Lock the frame pages with mlock
     * @param zero_new_frames
     */
    Buffer(int num_frames, size_t len, std::shared_ptr<metadataPool> pool,
           const std::string& buffer_name, const std::string& buffer_type, int numa_node,
           bool use_hugepages, bool mlock_frames, bool zero_new_frames);
    ~Buffer() override;

    /**
     * @brief Unregisters the named consumer from this buffer.
     * Signals producers if this causes the buffer to become free for
     * writing.
     */
    void unregister_consumer(const std::string& name) override;

    /**
     * @brief Prints a summary the frames and state of the producers and consumers.
     */
    void print_full_status() override;

    /**
     * @brief Zero all frames after all consumers have marked them as empty
     */
    void zero_frames();

    /**
     * @brief Marks a buffer frame as full.
     *
     * This function is used by a producer to sign off that it will no longer write
     * data to this frame.
     *
     * @param[in] producer_name The name of the producer registered with @c register_producer()
     * @param[in] frame_id The frame ID to be marked as full
     */
    void mark_frame_full(const std::string& producer_name, const int frame_id);

    /**
     * @brief Marks a buffer frame as empty
     *
     * Used by a consumer to sign off that it will no longer read data from this frame.
     *
     * @param[in] consumer_name The name of the consumer registered with @c register_consumer()
     * @param[in] frame_id The frame ID to be marked as empty
     */
    void mark_frame_empty(const std::string& consumer_name, const int frame_id);

    /**
     * @brief Blocks until the frame requested by frame_id is empty.
     *
     * This blocking function will return only when the frame_id request is marked
     * as empty internally, or the function @c send_shutdown_signal() is called, which
     * causes the function to return a @c NULL pointer.
     * Generally a stage should exit and cleanup if NULL is returned.
     *
     * @param[in] producer_name The name of the registered producer requesting the frame_id
     * @param[in] frame_id The id of the frame to wait for.
     * @returns A pointer to the frame, or NULL if the buffer is shutting down.
     * @warning After calling this function for a given producer and frame_id it
     *          should not be called again on that frame_id until after
     *          a call to @c mark_frame_full() with that producer and frame_id
     */
    uint8_t* wait_for_empty_frame(const std::string& producer_name, const int frame_id);

    /**
     * @brief Blocks until the frame requested by frame_id is full.
     *
     * This blocking function will return only when the frame_id request is marked
     * as full internally, or the function @c send_shutdown_signal() is called, which
     * causes the function to return a @c NULL pointer.
     * Generally a stage should exit and cleanup if NULL is returned.
     *
     * @param[in] consumer_name The name of the registered producer requesting the frame_id
     * @param[in] frame_id The id of the frame to wait for.
     * @returns A pointer to the frame, or NULL if the buffer is shutting down.
     * @warning After calling this function for a given consumer and frame_id it
     *          should not be called again on that frame_id until after
     *          a call to @c mark_frame_empty() with that consumer and frame_id
     */
    uint8_t* wait_for_full_frame(const std::string& consumer_name, const int frame_id);

    /**
     * @brief Wait for a full frame on the given buffer up to timeout.
     *
     * This function will timeout after `wait` seconds.
     *
     * @param[in] name Name of the stage.
     * @param[in] ID Frame ID to wait at.
     * @param[in] timeout Exit after this we exceed this *absolute* time.
     *
     * @return Return status:
     *   - `0`: Success! We have a new frame.
     *   - `1`: Failure! We timed out waiting.
     *   - `-1`: Failure! We received the thread exit signal.
     **/
    int wait_for_full_frame_timeout(const std::string& name, const int ID,
                                    const struct timespec timeout);

    /**
     * @brief Checks if the requested buffer is empty.
     *
     * Returns true if the buffer is empty, and false if the frame is full.
     *
     * @param[in] frame_id The id of the frame to check.
     * @warning This should not be used to gain access to an empty frame, use @c
     * wait_for_empty_frame()
     */
    bool is_frame_empty(const int frame_id);

    /**
     * @brief Returns the number of currently full frames.
     *
     * @returns The number of currently full frames in the buffer
     */
    int get_num_full_frames();

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
     * @param frame_id The frame to swap
     * @param external_frame The extra frame to use in place of the existing internal frame.
     * @return The internal frame
     */
    uint8_t* swap_external_frame(int frame_id, uint8_t* external_frame);

    /**
     * @brief Swaps frames between two buffers with identical size for the given frame_ids
     *
     * This function does not swap metadata.  That should be passed with the @c pass_metadata
     * function
     *
     * @warning This function should only be used with a single consumer @c from_buf, and given to a
     *          single producer @c to_buf.
     * @warning The buffer sizes must be identical.
     * @warning Take care with this function!
     *
     * @param from_frame_id The frame ID to move to the @c to_buf
     * @param to_buf The buffer to take the frame from @c from_buf
     * @param to_frame_id The frame to replace with the frame from @c from_buf
     */
    void swap_frames(int from_frame_id, Buffer* to_buf, int to_frame_id);

    /**
     * @brief Swaps a frame or performs a deep copy depending on the number of consumers on the
     *        source buffer.
     *
     * Like @c swap_frames(), but doesn't fail if there is more than one consumer on the source
     * buffer. Does not pass or copy metadata.
     *
     * @param[in] src_frame_id The source frame ID
     * @param[in] dest_buf The destination buffer
     * @param[in] dest_frame_id The destination frame ID
     */
    void safe_swap_frame(int src_frame_id, Buffer* dest_buf, int dest_frame_id);

    /**
     * @brief Returns the last time a frame was marked as full
     * @param buf The buffer to get the last arrival time for.
     * @return A double (with units: seconds) containing the unix time of the last frame arrival
     */
    double get_last_arrival_time();

    /**
     * @brief Prints a picture of the frames which are currently full.
     */
    void print_buffer_status() override;

    void json_description(nlohmann::json& buf_json) override;

    std::string get_dot_node_label() override;

    // don't call this, it's for internal use only
    void _impl_zero_frame(const int ID);

    // protected:
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

    /// Flag set to indicate if the frames should be zeroed between uses
    bool _zero_frames;

    /// The array of frames (the actual data we are carrying)
    std::vector<uint8_t*> frames;

    /**
     * @brief Flag variables to say which frames are full
     * A false at index I means the frame at index I is not full, true means it is full.
     */
    std::vector<bool> is_full;

    /// The last time a frame was marked as full (used for arrival rate)
    double last_arrival_time;

    /// This buffer use huge pages for its frames if the following is true
    bool use_hugepages;

    /// The buffer has page locked memory frames
    bool mlock_frames;

    /// The NUMA node the frames are allocated in
    int numa_node;

private:
    void private_mark_producer_done(const std::string& name, const int ID);
    // Returns true if all producers are done for the given ID.
    bool private_producers_done(const int ID);
    // Resets the list of producers for the given ID
    void private_reset_producers(const int ID);
    // Returns true if all consumers are done for the given ID.
    bool private_consumers_done(const int ID);
    /**
     * @brief Marks a frame as empty and if the buffer requires zeroing then it starts
     *        the zeroing thread and delays marking it as empty until the zeroing is done.
     * @param id The id of the frame to mark as empty.
     * @return True if the frame was marked as empty, false if it is being zeroed.
     */
    bool private_mark_frame_empty(const int ID);
    // Resets the list of consumers for the given ID
    void private_reset_consumers(const int ID);

    void private_copy_frame(int dest_frame_id, Buffer* src, int src_frame_id);
};

/**
 * @brief Returns true if the given GenericBuffer is a Buffer.
 */
bool is_frame_buffer(GenericBuffer* buf);

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

#endif
