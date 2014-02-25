#include "buffers.h"
#include "errors.h"

#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <math.h>
#include </opt/AMDAPP/include/CL/cl_platform.h>

/**
 * @brief private function to get the id of the first full buffer.
 * This should only be called from within a lock.
 */
int private_getFullBuffer(struct Buffer * buf);

/**
 * @brief finds a full buffer within a list of buffers, and returns its ID if it exists.
 * This should only be called from within a lock.
 */
int private_getFullBufferFromList(struct Buffer * buf, const int* buffer_IDs, const int len);

int private_areProducersDone(struct Buffer * buf);

int createBuffer(struct Buffer* buf, int num_buf, int len, int num_producers)
{

    assert(num_buf > 0);

    CHECK_ERROR( pthread_mutex_init(&buf->lock, NULL) );
    CHECK_ERROR( pthread_mutex_init(&buf->lock_info, NULL) );

    CHECK_ERROR( pthread_cond_init(&buf->full_cond, NULL) );
    CHECK_ERROR( pthread_cond_init(&buf->empty_cond, NULL) );

    buf->num_buffers = num_buf;
    // We need to align the buffer length to a page.
    buf->buffer_size = PAGESIZE_MEM * (ceil((double)len / (double)PAGESIZE_MEM));
    buf->num_producers = num_producers;

    // Create the is_free array
    buf->is_full = malloc(num_buf * sizeof(int));

    if ( buf->is_full == NULL ) {
        ERROR("Error creating is_full array");
        return errno;
    }

    memset(buf->is_full, 0, num_buf*sizeof(int));

    // Create the array of buffer pointers.
    buf->data = malloc(num_buf * sizeof(void *));

    if ( buf->data == NULL ) {
        ERROR("Error creating is_free array");
        return errno;
    }

    // Create the info array
    buf->info = malloc(num_buf*sizeof(struct BufferInfo));

    if ( buf->info == NULL ) {
        ERROR("Error creating info array");
        return errno;
    }

    memset(buf->is_full, 0, num_buf*sizeof(struct BufferInfo));

    // Create the array for tracking producers
    buf->producer_done = malloc(num_producers * sizeof(int));
    CHECK_MEM(buf->producer_done);

    memset(buf->producer_done, 0, num_producers * sizeof(int));

    int err = 0;

    // Create the actual buffers.
    for (int i = 0; i < num_buf; ++i) {

        // Create a page alligned block of memory for the buffer
        err = posix_memalign((void **) &(buf->data[i]), PAGESIZE_MEM, len);

        if ( err != 0 ) {
            ERROR("Error creating alligned memory");
            return err;
        }

        // Ask that all pages be kept in memory
        err = mlock((void *) buf->data[i], len);

        if ( err == -1 ) {
            ERROR("Error locking memory - check ulimit -a to check memlock limits");
            return errno;
        }
    }

    return 0;
}

void deleteBuffer(struct Buffer* buf)
{
    for (int i = 0; i < buf->num_buffers; ++i) {
        free(buf->data[i]);
    }

    free(buf->data);

    free(buf->is_full);

    free(buf->info);

    free(buf->producer_done);

    // Free locks and cond vars
    CHECK_ERROR( pthread_mutex_destroy(&buf->lock_info) );
    CHECK_ERROR( pthread_mutex_destroy(&buf->lock) );
    CHECK_ERROR( pthread_cond_destroy(&buf->full_cond) );
    CHECK_ERROR( pthread_cond_destroy(&buf->empty_cond) );
}

void markBufferFull(struct Buffer * buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->is_full[ID] = 1;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumer
    CHECK_ERROR( pthread_cond_broadcast(&buf->full_cond) );
}

int getFullBufferID(struct Buffer * buf)
{
    int fullBuf = -1;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    while ( private_getFullBuffer(buf) == -1 
            && private_areProducersDone(buf) == 0) {
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }

    fullBuf = private_getFullBuffer(buf);

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    return fullBuf;
}

void markBufferEmpty(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->is_full[ID] = 0;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal producer
    CHECK_ERROR( pthread_cond_broadcast(&buf->empty_cond) );
}

void waitForEmptyBuffer(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    // If the buffer isn't full, i.e. is_full[ID] == 0, then we never sleep on the cond var.
    while (buf->is_full[ID] == 1) {
        pthread_cond_wait(&buf->empty_cond, &buf->lock);
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
}


int is_buffer_empty(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);
    assert (buf != NULL);
    
    int empty = 1;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    if (buf->is_full[ID] == 1) {
        empty = 0;
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
    
    return empty;
}

int32_t get_buffer_data_ID(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    int dataID = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    dataID = buf->info[ID].data_ID;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return dataID;
}

uint32_t get_fpga_seq_num(struct Buffer* buf, const int ID)
{
    int fpga_seq_num = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    fpga_seq_num = buf->info[ID].fpga_seq_num;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return fpga_seq_num;
}

struct timeval get_first_packet_recv_time(struct Buffer* buf, const int ID)
{
    struct timeval time;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    time = buf->info[ID].first_packet_recv_time;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return time;
}

void setDataID(struct Buffer* buf, const int ID, const int data_ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    buf->info[ID].data_ID = data_ID;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}

void set_fpga_seq_num(struct Buffer* buf, const int ID, const uint32_t fpga_seq_num)
{
    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    buf->info[ID].fpga_seq_num = fpga_seq_num;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}

void set_first_packet_recv_time(struct Buffer* buf, const int ID, struct timeval time)
{
    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    buf->info[ID].first_packet_recv_time = time;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}


int getFullBufferFromList(struct Buffer* buf, const int* buffer_IDs, const int len)
{
    int fullBuf = -1;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    while ( private_getFullBufferFromList(buf, buffer_IDs, len) == -1 
            && private_areProducersDone(buf) == 0 ) {
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }

    fullBuf = private_getFullBufferFromList(buf, buffer_IDs, len);

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    return fullBuf;
}

void markProducerDone(struct Buffer* buf, int producer_id)
{
    assert(buf != NULL);
    assert(producer_id >= 0);
    assert(producer_id < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->producer_done[producer_id] = 1;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumers
    CHECK_ERROR( pthread_cond_broadcast(&buf->full_cond) );
}

int getNumFullBuffers(struct Buffer* buf)
{
    int numFull = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    for (int i = 0; i < buf->num_buffers; ++i) {
        if (buf->is_full[i] == 1) {
            numFull++;
        }
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    return numFull;
}

void printBufferStatus(struct Buffer* buf)
{
    int is_full[buf->num_buffers];

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    memcpy(is_full, buf->is_full, buf->num_buffers * sizeof(int));

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    char status_string[buf->num_buffers];

    for (int i = 0; i < buf->num_buffers; ++i) {
        if (buf->is_full[i] == 1) {
            status_string[i] = 'X';
        } else {
            status_string[i] = '_';
        }
    }
    DEBUG("Buffer Status: %s", status_string);
}

void copy_buffer_info(struct Buffer * from, int from_id, struct Buffer * to, int to_id)
{
    assert(from != NULL);
    assert(to != NULL);

    CHECK_ERROR( pthread_mutex_lock(&from->lock_info) );
    CHECK_ERROR( pthread_mutex_lock(&to->lock_info) );

    memcpy((void *)&to->info[to_id], (void *)&from->info[from_id], sizeof(struct BufferInfo));

    CHECK_ERROR( pthread_mutex_unlock(&to->lock_info) );
    CHECK_ERROR( pthread_mutex_unlock(&from->lock_info) );
}

int private_getFullBuffer(struct Buffer* buf)
{
    for (int i = 0; i < buf->num_buffers; ++i) {
        if (buf->is_full[i] == 1) {
            return i;
        }
    }
    return -1;
}

int private_getFullBufferFromList(struct Buffer* buf, const int* buffer_IDs, const int len)
{
    for(int i = 0; i < len; ++i) {
        assert (buffer_IDs[i] >= 0);
        assert (buffer_IDs[i] < buf->num_buffers);

        if (buf->is_full[buffer_IDs[i]] == 1) {
            return buffer_IDs[i];
        }
    }
    return -1;
}

int private_areProducersDone(struct Buffer* buf)
{
    // Assume we are done.
    int result = 1;
    for (int i = 0; i < buf->num_producers; ++i) {
        if (buf->producer_done[i] == 0) {
            result = 0;
            break;
        }
    }
    return result;
}


