/**
 * @file buffer.h
 * @brief The buffer object used by each link and the OpenCL thread for buffering data
 *        from the network, and transfering to the GPU.  Most functions here are thread safe.
 */

#ifndef BUFFER
#define BUFFER

#ifdef __cplusplus
extern "C" {
#endif

#define PAGESIZE_MEM 4096

#define MAX_PROCESS_NAME_LEN 128
// 10 Should be enough for most situations
#define MAX_CONSUMERS 10
#define MAX_PRODUCERS 10

#include <pthread.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef MAC_OSX
#include "osxBindCPU.hpp"
#include <immintrin.h>
#endif

#include "metadata.h"

struct ProcessInfo {
    int in_use;
    char name[MAX_PROCESS_NAME_LEN];
};

/** @brief Buffer object used to contain and manage the buffers shared by the network
 *  and consumer (OpenCL, file reading) threads.
 *
 * All memory here is shared between threads so it must be used with the associated locks
 */
struct Buffer {

    /// Lock and cond variable.
    pthread_mutex_t lock;
    pthread_cond_t full_cond;
    pthread_cond_t empty_cond;

    /// Shutdown variable set to 1 when the system should stop returning
    /// new frames for producers and consumers
    int shutdown_signal;

    /// The number of frames kept by this object
    int num_frames;

    /// The size of each frame.
    int frame_size;

    // Each frame is padded out to a page aligned size.
    int aligned_frame_size;

    /// Array of producers which are done (marked frame as full).
    /// [ID][producer]
    /// zero means not done, 1 means done (marked as full)
    int ** producers_done;

    /// Array of consumers which are done (marked frame as empty).
    /// [ID][consumer]
    /// zero means not done, 1 means done (marked as empty))
    int ** consumers_done;

    struct ProcessInfo consumers[MAX_CONSUMERS];
    struct ProcessInfo producers[MAX_PRODUCERS];

    /// Should be frames be zeroed at the end of its use.
    int zero_frames;

    /// The array of buffers
    uint8_t ** frames;

    /// Flag vars to say which frames are full
    /// A 0 at index I means the frame at index I is not full, one means it is full.
    int * is_full;

    /// Array of buffer info objects, for tracking information about each buffer.
    struct metadataContainer ** metadata;

    /// The pool of info objects
    struct metadataPool * metadata_pool;

    /// The name of the buffer for use in debug messages.
    char * buffer_name;
};

/** @brief Creates a buffer object.
 *  Not thread safe.
 *  @param [in] num_buf The number of buffers to create in the buffer object.
 *  @param [in] len The lenght of each buffer to be created in bytes.
 *  @param [in] pool The metadataPool, which may be shared between more than one buffer.
 *  @param [in] buffer_name The name of this buffer.
 *  @return A buffer object.
 */
struct Buffer * create_buffer(int num_frames, int len,
                  struct metadataPool * pool, const char * buffer_name);

// Calling this function makes the buffer zero frames after each use.
void zero_frames(struct Buffer * buf);

/** @brief Deletes a buffer object
 *  Not thread safe.
 *  @param [in] buf The buffer to delete.
 *  @return 0 if successful, or a non-zero standard error value if not.
 */
void delete_buffer(struct Buffer * buf);

void register_consumer(struct Buffer * buf, const char *name);

void register_producer(struct Buffer * buf, const char *name);

/** @brief Mark a buffer as full.
 *  This function is thread safe.
 *  @param ID The id of the frame to mark as full.
 */
void mark_frame_full(struct Buffer * buf, const char * producer_name, const int ID);

/** @Brief Marks a buffer as empty
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] ID The id of the frame to mark as empty.
 */


void mark_frame_empty(struct Buffer* buf, const char * consumer_name, const int ID);

/**
 * Blocks until the frame requested by frame_id is empty.
 *
 * This blocking function will return only when the frame_id request is marked
 * as empty internally, or the function @c send_shutdown_signal() is called, which
 * causes the function to return a @c NULL pointer.
 *
 * @param[in] buf The buffer object
 * @param[in] producer_name The name of the registered producer requesting the frame_id
 * @param[in] frame_id The id of the frame wait for.
 * @returns A pointer to the frame, or NULL if the buffer is shutting down.
 * @warning You may only make this call once per registered producer.
 */
uint8_t * wait_for_empty_frame(struct Buffer* buf, const char * producer_name, const int frame_id);

/** @brief Waits for the buffer frame given by ID to be full.
 *  This function is thread safe.
 */
uint8_t * wait_for_full_frame(struct Buffer* buf, const char * consumer_name, const int ID);

/** @brief Checks if the requested buffer is empty, returns 1
 *  if the buffer is empty, and 0 if the full.  Thread safe.
 * @param [in] buf The buffer
 * @param [in] ID The id of the frame to check.
 */
int is_frame_empty(struct Buffer * buf, const int ID);

/**
 * @brief Returns the number of currently full frames.
 */
int get_num_full_frames(struct Buffer * buf);

/**
 * @brief Prints a picture of the frames which are currently full.
 */
void print_buffer_status(struct Buffer * buf);

// Needs to be called by the first producer in a chain, or by a producing generating
// a new type of metadata for the next stage.
void allocate_new_metadata_object(struct Buffer * buf, int ID);

// Just gets the raw metadata block, which can be cast as needed into
// the required metadata type.
void * get_metadata(struct Buffer * buf, int ID);

struct metadataContainer * get_metadata_container(struct Buffer * buf, int ID);

// Must be called before marking the from buffer as empty.
void pass_metadata(struct Buffer * from_buf, int from_ID, struct Buffer * to_buf, int to_ID);

// Causes all "wait" commands to wakeup and return NULL
void send_shutdown_signal(struct Buffer * buf);

#ifdef __cplusplus
}
#endif

#endif
