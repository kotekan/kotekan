/**
 * @file buffer.h
 * @brief The buffer object used by each link and the OpenCL thread for buffering data
 *        from the network, and transfering to the GPU.  Most functions here are thread safe.
 */

#ifndef BUFFERS
#define BUFFERS

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

#include "error_correction.h"

/** @brief Object containing information about a given buffer.
 *
 */
struct BufferInfo {

    /// Buffer data_ID.
    /// This is the ID of the data in the buffer, not the ID of the buffer.
    int32_t data_ID;
    int64_t fpga_seq_num;
    struct timeval first_packet_recv_time;
    struct ErrorMatrix error_matrix;
    uint16_t stream_ID;

    // 1 = in use, 0 = avaiable;
    char in_use;

    int ref_count;
};

struct InfoObjectPool {
    struct BufferInfo * info_objects;

    unsigned int pool_size;

    // Always lock when changing the in_use status of an info object.
    pthread_mutex_t in_use_lock;
};

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
    pthread_mutex_t lock_info;  // The lock for the info struct.
    pthread_cond_t full_cond;
    pthread_cond_t empty_cond;

    /// The number of buffers kept by this object
    int num_buffers;

    /// The size of each buffer.
    int buffer_size;

    // Each buffer is padded out to a page aligned size.
    int aligned_buffer_size;

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

    /// Should be buffer be zeroed at the end of its use.
    int * zero_buffer;

    /// The array of buffers
    unsigned char ** data;

    /// Flag vars to say which buffers are full
    /// A 0 at index I means the buffer at index I is not full, one means it is full.
    int * is_full;

    /// Array of buffer info objects, for tracking information about each buffer.
    struct BufferInfo ** info;

    /// The pool of info objects
    struct InfoObjectPool * info_object_pool;

    /// The name of the buffer for use in debug messages.
    char * buffer_name;
};

/** @brief Creates a buffer object.
 *  Not thread safe.
 *  @param [out] buf A pointer to a buffer object to be initialized.
 *  @param [in] num_buf The number of buffers to create in the buffer object.
 *  @param [in] len The lenght of each buffer to be created in bytes.
 *  @param [in] pool The BufferInfo object pool, which may be shared between more than one buffer.
 *  @return 0 if successful, or a non-zero standard error value if not successful
 */
int create_buffer(struct Buffer * buf, int num_buf, int len,
                  struct InfoObjectPool * pool, const char * buffer_name);

/** @brief Deletes a buffer object
 *  Not thread safe.
 *  @param [in] buf The buffer to delete.
 *  @return 0 if successful, or a non-zero standard error value if not.
 */
void delete_buffer(struct Buffer * buf);

void register_consumer(struct Buffer * buf, const char *name);

void register_producer(struct Buffer * buf, const char *name);

// Returns -1 if there is no consumer with that name
int  private_get_consumer_id(struct Buffer * buf, const char * name);

// Returns -1 if there is no producer with that name
int  private_get_producer_id(struct Buffer * buf, const char * name);

// Marks the consumer named by `name` as done for the given ID
void private_mark_consumer_done(struct Buffer * buf, const char * name, const int ID);

// Marks the producer named by `name` as done for the given ID
void private_mark_producer_done(struct Buffer * buf, const char * name, const int ID);

// Returns 1 if all consumers are done for the given ID.
int  private_consumers_done(struct Buffer * buf, const int ID);

// Returns 1 if all producers are done for the given ID.
int  private_producers_done(struct Buffer * buf, const int ID);

// Resets the list of producers for the given ID
void private_reset_producers(struct Buffer * buf, const int ID);

// Resets the list of consumers for the given ID
void private_reset_consumers(struct Buffer * buf, const int ID);

struct zero_buffer_thread_args {
    struct Buffer * buf;
    int ID;
};
void *private_zero_buffer(void * args);

void zero_buffer(struct Buffer * buf, const int ID);

/** @brief Gets ID of a buffer which is full.
 *  This function is thread safe, and will block if not buffers are free.
 *  @return The ID of a buffer which is full. Or -1 if the producer is done filling buffers.
 */
int get_full_buffer_ID(struct Buffer * buf);

/** @brief Mark a buffer as full.
 *  This function is thread safe.
 *  @param ID The id of the buffer to mark as full.
 */
void mark_buffer_full(struct Buffer * buf, const char * producer_name, const int ID);

/** @brief Waits for the buffer frame given by ID to be full.
 *  This function is thread safe.
 */
int wait_for_full_buffer(struct Buffer* buf, const char * consumer_name, const int ID);

/** @brief Gets the data_ID of the buffer with the given ID.
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] ID The ID of the buffer
 *  @return The data ID associated with the buffer.
 */
int32_t get_buffer_data_ID(struct Buffer * buf, const int ID);

uint64_t get_fpga_seq_num(struct Buffer * buf, const int ID);

int32_t get_streamID(struct Buffer * buf, const int ID);

struct timeval get_first_packet_recv_time(struct Buffer * buf, const int ID);

// TODO/HACK This function bypasses the thread safety systems that the other get_ functions have.
// This is not ideal, although in the current implementation of the network/gpu/file_write
// code, we do not actually require thread safety here.
// Having said that, either all get_ functions should lose thread safety, or this one
// should get thread safety for consistency - by creating proxy functions which are safe.
struct ErrorMatrix * get_error_matrix(struct Buffer * buf, const int ID);

/** @brief Sets the data_ID for a buffer.
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] ID The ID of the buffer
 *  @param [in] data_ID The data ID of the data in the buffer.
 */
void set_data_ID(struct Buffer * buf, const int ID, const int data_ID);

void set_fpga_seq_num(struct Buffer * buf, const int ID, const uint64_t fpga_seq_num);

void set_stream_ID(struct Buffer * buf, const int ID, const uint16_t stream_ID);

void set_first_packet_recv_time(struct Buffer * buf, const int ID, const struct timeval time);

/** @Brief Marks a buffer as empty
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] ID The id of the buffer to mark as empty.
 */
void mark_buffer_empty(struct Buffer* buf, const char * consumer_name, const int ID);

/** @brief Blocks until the buffer requested is empty.
 *  This function is thread safe.
 *  @param [in] buf The buffer
 *  @param [in] ID The id of the buffer wait for.
 */
void wait_for_empty_buffer(struct Buffer* buf, const char * producer_name, const int ID);

/** @brief Checks if the requested buffer is empty, returns 1
 *  if the buffer is empty, and 0 if the full.  Thread safe.
 * @param [in] buf The buffer
 * @param [in] ID The id of the buffer to check.
 */
int is_buffer_empty(struct Buffer * buf, const int ID);

/**
 * @brief Tells the buffer that no new data is coming.
 * Consumer threads are free to exit as soon as all buffers are empty.
 */
void mark_producer_done(struct Buffer * buf, int producer_id);

/**
 * @brief Returns the number of currently full buffers.
 */
int get_num_full_buffers(struct Buffer * buf);

/**
 * @brief Prints a picture of the buffers which are currently full.
 */
void print_buffer_status(struct Buffer * buf);

void move_buffer_info(struct Buffer * from, int from_id, struct Buffer * to, int to_id);

void copy_buffer_info(struct Buffer * from, int from_id, struct Buffer * to, int to_id);

void release_info_object(struct Buffer * buf, const int ID);

void create_info_pool(struct InfoObjectPool * pool, int num_info_objects, int num_freq, int num_elem);

struct BufferInfo * request_info_object(struct InfoObjectPool * pool);

void return_info_object(struct InfoObjectPool * pool, struct BufferInfo * buffer_info);

void reset_info_object(struct BufferInfo * buffer_info);

void delete_info_object_pool(struct InfoObjectPool * pool);

#ifdef __cplusplus
}
#endif

#endif
