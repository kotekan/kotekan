/**
 * @file buffer.h
 * @brief The core kotekan buffer object for data transfer between processes
 *  - buffer
 *  - ProcessInfo
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

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef MAC_OSX
#include "osxBindCPU.hpp"
#include <immintrin.h>
#endif

#include "metadata.h"

/// The system page size, this might become more dynamic someday
#define PAGESIZE_MEM 4096

/// The max length of a process (consumer or producer) name.
#define MAX_PROCESS_NAME_LEN 128

/// The maximum number of consumers that can register on a buffer
#define MAX_CONSUMERS 10
/// The maximum number of producers that can register on a buffer
#define MAX_PRODUCERS 10

/**
 * @struct ProcessInfo
 * @brief Internal structure for tracking consumer and producer names.
 */
struct ProcessInfo {

    /// Set to 1 if the process is active
    int in_use;

    /// The name of the process (consumer or producer)
    char name[MAX_PROCESS_NAME_LEN];
};

/**
 * @struct Buffer
 * @brief Kotekan's core multi-producer, multi-consumer ring buffer with metadata
 *
 * This class is the central method for passing data between kotekan processes
 * in a pipeline.
 *
 * It provides a fixed size RING buffer which can have multiple producers and
 * consumers attached to it.   The idea is that individual processes do not need
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
 * Consumers must only read data from frames and not write anything back too them.
 * More than one producer can write to a given frame in a multi producer setup,
 * but in that case they must coordinate their address space to not overwrite
 * each others values.  Because of this multi-producers are somewhat rare.
 * Producers also generally shouldn't read from frames, although there is
 * nothing wrong with doing so, it just normally doesn't make sense to do so.
 *
 * Unless the function @c zero_frames() is called on the buffer object, the
 * default behaviour is not to zero the memory of the frames between uses.
 * Therefore it is normally upto the producer(s) to ensure all memory
 * values are either given new data, or zeroed.
 *
 * In the config file a buffer is created with a <tt>kotekan_buffer: standard</tt>
 * named block.   The buffer name becomes the path name of that config block.
 *
 * Note if no consumer is registered for on a buffer, then it will drop
 * the frames and log an INFO statement to notify the user that the data
 * is being dropped.
 *
 * @conf frame_size The size of the individual ring frames in bytes
 * @conf num_frames The buffer depth of size of the ring
 * @conf metadata_pool The name of the metadata pool to associate with the buffer
 *
 * See metadata.h for more information on metadata pools
 *
 * @author Andre Renard
 */
struct Buffer {

    /// The main lock for frame state management
    pthread_mutex_t lock;

    /// The condition variable for calls to @c wait_for_full_buffer
    pthread_cond_t full_cond;

    /// The condition variable for calls to @c wait_for_empty_buffer
    pthread_cond_t empty_cond;

    /**
     * @brief Shutdown variable
     * Set to 1 when the system should stop returning
     * new frames for producers and consumers.
     */
    int shutdown_signal;

    /// The number of frames kept by this object
    int num_frames;

    /// The size of each frame in bytes.
    int frame_size;

    /**
     * @brief The padded frame size.
     * Each frame is padded out to a page aligned size,
     * your operation wants to do things paged aligned
     * you can use this size instead, but data shouldn't
     * be placed past the end of frame_size.  This is just for padding.
     */
    int aligned_frame_size;

    /**
     * @brief Array of producers which are done (marked frame as full).
     * Format is [ID][producer]
     * zero means not done, 1 means done (marked as full)
     */
    int ** producers_done;

    /**
     * @brief Array of consumers which are done (marked frame as empty).
     * Format is [ID][consumer]
     * zero means not done, 1 means done (marked as empty)
     */
    int ** consumers_done;

    /// The list of consumer names registered to this buffer
    struct ProcessInfo consumers[MAX_CONSUMERS];

    /// The list of producer names registered to this buffer
    struct ProcessInfo producers[MAX_PRODUCERS];

    /// Flag set to indicate if the frames should be zeroed between uses
    int zero_frames;

    /// The array of frames (the actual data we are carrying)
    uint8_t ** frames;

    /**
     * @brief Flag variables to say which frames are full
     * A 0 at index I means the frame at index I is not full, one means it is full.
     */
    int * is_full;

    /// Array of buffer info objects, for tracking information about each buffer.
    struct metadataContainer ** metadata;

    /// The pool of info objects
    struct metadataPool * metadata_pool;

    /// The name of the buffer for use in debug messages.
    char * buffer_name;
};

/**
 * @brief Creates a buffer object.
 *
 * Used to create a buffer object, normally invoked by the buffer factory
 * as a part of the pipeline generation from the config file, not intended
 * to be called directly.
 *
 * @param[in] num_frame The number of frames to create in the buffer ring.
 * @param[in] frame_size The length of each frame in bytes.
 * @param[in] pool The metadataPool, which may be shared between more than one buffer.
 * @param[in] buffer_name The unique name of this buffer.
 * @returns A buffer object.
 */
struct Buffer * create_buffer(int num_frames, int frame_size,
                  struct metadataPool * pool, const char * buffer_name);

/**
 * @brief Deletes a buffer object and frees all frame memory
 *
 * @param[in] buf The buffer to delete.
 */
void delete_buffer(struct Buffer * buf);

/**
 * @brief Zero all frames after all consumers have marked them as empty
 *
 * @param[in] buf The buffer object which will be set to automatically zero all frames
 */
void zero_frames(struct Buffer * buf);

/**
 * @brief Register a consumer with a given name.
 *
 * In order to use a buffer a consumer must first register its name so that
 * the buffer object can track which consumers have signed off on each frame.
 *
 * @param[in] buf The buffer to register on
 * @param[in] name The name of the consumer.
 */
void register_consumer(struct Buffer * buf, const char *name);

/**
 * @brief Register a producer with a given name.
 *
 * In order to use a buffer a producer must first register its name so that
 * the buffer object can track which producers have signed off on each frame.
 *
 * @param[in] buf The buffer to register on
 * @param[in] name The name of the producer.
 */
void register_producer(struct Buffer * buf, const char *name);

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
void mark_frame_full(struct Buffer * buf, const char * producer_name, const int frame_id);

/**
 * @brief Marks a buffer frame as empty
 *
 * Used by a consumer to sign off that it will no longer read data from this frame.
 *
 * @param[in] buf The buffer containing the frame to mark as empty
 * @param[in] consumer_name The name of the consumer registered with @c register_consumer()
 * @param[in] frame_id The frame ID to be marked as empty
 */
void mark_frame_empty(struct Buffer* buf, const char * consumer_name, const int frame_id);

/**
 * @brief Blocks until the frame requested by frame_id is empty.
 *
 * This blocking function will return only when the frame_id request is marked
 * as empty internally, or the function @c send_shutdown_signal() is called, which
 * causes the function to return a @c NULL pointer.
 * Generally a process should exit and cleanup if NULL is returned.
 *
 * @param[in] buf The buffer object
 * @param[in] producer_name The name of the registered producer requesting the frame_id
 * @param[in] frame_id The id of the frame to wait for.
 * @returns A pointer to the frame, or NULL if the buffer is shutting down.
 * @warning You may only make this call once per registered producer.
 */
uint8_t * wait_for_empty_frame(struct Buffer* buf, const char * producer_name, const int frame_id);

/**
 * @brief Blocks until the frame requested by frame_id is full.
 *
 * This blocking function will return only when the frame_id request is marked
 * as full internally, or the function @c send_shutdown_signal() is called, which
 * causes the function to return a @c NULL pointer.
 * Generally a process should exit and cleanup if NULL is returned.
 *
 * @param[in] buf The buffer object
 * @param[in] producer_name The name of the registered producer requesting the frame_id
 * @param[in] frame_id The id of the frame to wait for.
 * @returns A pointer to the frame, or NULL if the buffer is shutting down.
 * @warning You may only make this call once per registered producer.
 */
uint8_t * wait_for_full_frame(struct Buffer* buf, const char * consumer_name, const int frame_id);

/**
 * @brief Checks if the requested buffer is empty.
 *
 * Returns 1 if the buffer is empty, and 0 if the frame is full.
 *
 * @param[in] buf The buffer object
 * @param[in] frame_id The id of the frame to check.
 * @warning This should not be used to gain access to an empty frame, use @c wait_for_empty_frame()
 */
int is_frame_empty(struct Buffer * buf, const int frame_id);

/**
 * @brief Returns the number of currently full frames.
 *
 * @param[in] buf The buffer object
 * @returns The number of currently full frames in the buffer
 */
int get_num_full_frames(struct Buffer * buf);

/**
 * @brief Prints a picture of the frames which are currently full.
 *
 * @param[in] buf The buffer object
 */
void print_buffer_status(struct Buffer * buf);

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
void allocate_new_metadata_object(struct Buffer * buf, int frame_id);

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
void * get_metadata(struct Buffer * buf, int frame_id);

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
struct metadataContainer * get_metadata_container(struct Buffer * buf, int frame_id);

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
 * access to this metadata object.  The process releases the metadata implicitly
 * when the @c mark_frame_empty() function is called, which decrements
 * the reference counter. Once it reaches zero, the the metadata is returned to the
 * pool.
 *
 * @param[in] from_buf The buffer to copy the metadata from
 * @param[in] from_frame_id The frame ID to copy the metadata from
 * @param[in] to_buf The buffer to copy the metadata into
 * @param[in] to_frame_id The frame ID in the @c to_buf to copy the metadata into
 */
void pass_metadata(struct Buffer * from_buf, int from_frame_id,
                    struct Buffer * to_buf, int to_frame_id);

/**
 * @brief Tells the buffers to stop returning full/empty frames to consumers/producers
 *
 * This function should only be called by the framework, and not by processes.
 * Once called it will cause all @c wait_for_empty_frame() and @c wait_for_full_frame()
 * calls to wake up and return NULL; or NULL on the next time they are called.
 *
 * @param[in] buf The buffer to shutdown
 */
void send_shutdown_signal(struct Buffer * buf);

#ifdef __cplusplus
}
#endif

#endif