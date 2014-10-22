/**
 * @file buffer.h
 * @brief The buffer object used by each link and the OpenCL thread for buffering data
 *        from the network, and transfering to the GPU.  Most functions here are thread safe.
 */

#ifndef BUFFERS
#define BUFFERS

#define PAGESIZE_MEM 4096

#include <pthread.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>

#define HANDLE_ERROR( err )                                         \
    if ( err ) {                                                    \
        printf("Error in %s:%d \n Error type: %s \n",               \
                __FILE__, __LINE__, strerror(errno));               \
        exit( errno );                                              \
    }

/** @brief Object containing information about a given buffer.
 *
 */
struct BufferInfo {

    /// Buffer data_ID. 
    /// This is the ID of the data in the buffer, not the ID of the buffer.
    int32_t data_ID;

    uint32_t fpga_seq_num;
    struct timeval first_packet_time;
};

/** @brief Buffer object used to contain and manage the buffers shared by the network 
 *  and consumer (OpenCL, file reading) threads.
 * 
 * All memory here is shared between threads so it must be used with the assoicated locks
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

    // Number of producers
    int num_producers;

    // Number of consumers
    int num_consumers;

    /// The array of buffers
    char ** data;

    /// Flag vars to say which buffers are full
    /// A 0 at index I means the buffer at index I is not full, one means it is full.
    int * is_full;

    /// The number of producers
    int * num_producers_done;

    /// The number of consumers done.
    int * num_consumers_done;

    /// The total number of free buffers, used by cond variable to check if there
    /// is space to write to.
    int num_free;

    /// Array of buffer info objects, for tracking information about each buffer.
    struct BufferInfo * info;

    /// Flag to say when the producer is finished adding data to the buffer.
    /// 0 means the producer is not done, 1 means the producer is done.
    int producer_done;
};

/** @brief Creates a buffer object.
 *  Not thread safe.
 *  @param [out] buf A pointer to a buffer object to be initialized.
 *  @param [in] num_buf The number of buffers to create in the buffer object.
 *  @param [in] len The lenght of each buffer to be created in bytes.
 *  @return 0 if successful, or a non-zero standard error value if not successful 
 */
int createBuffer(struct Buffer * buf, int num_buf, int len, int num_producers, int num_consumers);

/** @brief Deletes a buffer object
 *  Not thread safe.
 *  @param [in] buf The buffer to delete.
 *  @return 0 if successful, or a non-zero standard error value if not. 
 */
void deleteBuffer(struct Buffer * buf);

/** @brief Gets ID of a buffer which is full.
 *  This function is thread safe, and will block if not buffers are free.
 *  @return The ID of a buffer which is full. Or -1 if the producer is done filling buffers.
 */
int getFullBufferID(struct Buffer * buf);

/** @brief Mark a buffer as full.
 *  This function is thread safe.
 *  @param ID The id of the buffer to mark as full.
 */
void markBufferFull(struct Buffer * buf, const int ID);

void markBufferFull_nProducers(struct Buffer * buf, const int ID);

/** @brief Waits for one of the buffers given to be full.
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] buffer_IDs An array of buffer IDs to wait for.
 *  @param [in] len The lenght of the array of buffer IDs.
 *  @return The ID of the buffer that is full.  Or -1 if the producer is done filling buffers.
 */
int getFullBufferFromList(struct Buffer * buf, const int * buffer_IDs, const int len);

/** @brief Gets the data_ID of the buffer with the given ID.
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] ID The ID of the buffer
 *  @return The data_ID assoicated with the buffer.
 */
int getDataID(struct Buffer * buf, const int ID);

/** @brief Sets the data_ID for a buffer.
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] ID The ID of the buffer
 *  @param [in] data_ID The data ID of the data in the buffer.
 */
void setDataID(struct Buffer * buf, const int ID, const int data_ID);

/** @Brief Marks a buffer as empty
 *  This function is thread safe.
 *  @param [in] buf The buffer object
 *  @param [in] ID The id of the buffer to mark as empty.
 */
void markBufferEmpty(struct Buffer * buf, const int ID);

void markBufferEmpty_nConsumers(struct Buffer * buf, const int ID);

/** @brief Blocks until the buffer requested is empty.
 *  This function is thread safe.
 *  @param [in] buf The buffer
 *  @param [in] ID The id of the buffer wait for.
 */
void waitForEmptyBuffer(struct Buffer * buf, const int ID);

/**
 * @brief Tells the buffer that no new data is coming. 
 * Consumer threads are free to exit as soon as all buffers are empty.
 */
void markProducerDone(struct Buffer * buf);

/**
 * @brief Returns the number of currently full buffers.
 */
int getNumFullBuffers(struct Buffer * buf);

/**
 * @brief Prints a picture of the buffers which are currently full.
 */
void printBufferStatus(struct Buffer * buf);

#endif