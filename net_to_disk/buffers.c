#include "buffers.h"

#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

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

int createBuffer(struct Buffer* buf, int num_buf, int len, int num_producers, int num_consumers)
{

    assert(num_buf > 0);

    HANDLE_ERROR( pthread_mutex_init(&buf->lock, NULL) );
    HANDLE_ERROR( pthread_mutex_init(&buf->lock_info, NULL) );

    HANDLE_ERROR( pthread_cond_init(&buf->full_cond, NULL) );
    HANDLE_ERROR( pthread_cond_init(&buf->empty_cond, NULL) );

    buf->num_buffers = num_buf;
    buf->buffer_size = len;
    buf->producer_done = 0;
    buf->num_producers = num_producers;
    buf->num_consumers = num_consumers;

    // Create the is_free array
    buf->is_full = malloc(num_buf * sizeof(int));
    if ( buf->is_full == NULL ) {
        perror("Error creating is_full array");
        return errno;
    }
    memset(buf->is_full, 0, num_buf*sizeof(int));

    // Create the array of buffer pointers.
    buf->data = malloc(num_buf * sizeof(void *));
    if ( buf->data == NULL ) {
        perror("Error creating is_free array");
        return errno;
    }

    // Create the info array
    buf->info = malloc(num_buf*sizeof(struct BufferInfo));
    if ( buf->info == NULL ) {
        perror("Error creating info array");
        return errno;
    }
    memset(buf->info, 0, num_buf*sizeof(int));

    // Create the producers done array
    buf->num_producers_done = malloc(num_buf*sizeof(int));
    if ( buf->num_producers_done == NULL ) {
        perror("Error creating num_producers_done array");
        return errno;
    }
    memset(buf->num_producers_done, 0, num_buf*sizeof(int));

    // Create the producers done array
    buf->num_consumers_done = malloc(num_buf*sizeof(int));
    if ( buf->num_consumers_done == NULL ) {
        perror("Error creating num_consumers_done array");
        return errno;
    }
    memset(buf->num_consumers_done, 0, num_buf*sizeof(int));

    int err = 0;

    // Create the actual buffers.
    for (int i = 0; i < num_buf; ++i) {

        // Create a page alligned block of memory for the buffer
        err = posix_memalign((void **) &(buf->data[i]), PAGESIZE_MEM, len);

        if ( err != 0 ) {
            printf("Error creating alligned memory");
            return err;
        }

        // Ask that all pages be kept in memory
        err = mlock((void *) buf->data[i], len);

        if ( err == -1 ) {
            perror("Error locking memory - check ulimit -a to check memlock limits");
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
    free(buf->num_producers_done);

    free(buf->info);

    // Free locks and cond vars
    HANDLE_ERROR( pthread_mutex_destroy(&buf->lock_info) );
    HANDLE_ERROR( pthread_mutex_destroy(&buf->lock) );
    HANDLE_ERROR( pthread_cond_destroy(&buf->full_cond) );
    HANDLE_ERROR( pthread_cond_destroy(&buf->empty_cond) );
}

void markBufferFull(struct Buffer * buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->is_full[ID] = 1;

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumer
    HANDLE_ERROR( pthread_cond_broadcast(&buf->full_cond) );
}

void markBufferFull_nProducers(struct Buffer * buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    int set_full = 0;

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->num_producers_done[ID]++;
    if (buf->num_producers_done[ID] == buf->num_producers) {
        //fprintf(stderr, "Marking buffer as full\n");
        buf->is_full[ID] = 1;
        set_full = 1;
    }

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumer
    if (set_full == 1) {
        HANDLE_ERROR( pthread_cond_broadcast(&buf->full_cond) );
    }
}

int getFullBufferID(struct Buffer * buf)
{

    int fullBuf = -1;

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    while ( private_getFullBuffer(buf) == -1 && buf->producer_done == 0) {
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }

    fullBuf = private_getFullBuffer(buf);

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    return fullBuf;
}

void markBufferEmpty(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->is_full[ID] = 0;
    buf->num_producers_done[ID] = 0;

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal producer
    HANDLE_ERROR( pthread_cond_broadcast(&buf->empty_cond) );
}

void markBufferEmpty_nConsumers(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    int broadcast = 0;

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->num_consumers_done[ID]++;
    if (buf->num_consumers_done[ID] == buf->num_consumers) {
        buf->is_full[ID] = 0;
        buf->num_producers_done[ID] = 0;
        buf->num_consumers_done[ID] = 0;
        broadcast = 1;
    }

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal producer
    if (broadcast == 1) {
        HANDLE_ERROR( pthread_cond_broadcast(&buf->empty_cond) );
    }
}

void waitForEmptyBuffer(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    //fprintf(stderr, "num_buffers: %d; ID: %d", buf->num_buffers, ID);
    assert (ID < buf->num_buffers);

    //printf("Waiting for buffer %d\n", ID);
    //printBufferStatus(buf);

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    // If the buffer isn't full, i.e. is_full[ID] == 0, then we never sleep on the cond var.
    while (buf->is_full[ID] == 1) {
        pthread_cond_wait(&buf->empty_cond, &buf->lock);
    }

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );
}

int getDataID(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    int dataID = 0;

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock_info) );

    dataID = buf->info[ID].data_ID;

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return dataID;
}

void setDataID(struct Buffer* buf, const int ID, const int data_ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock_info) );

    buf->info[ID].data_ID = data_ID;

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}

int getFullBufferFromList(struct Buffer* buf, const int* buffer_IDs, const int len)
{
    int fullBuf = -1;

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    while ( private_getFullBufferFromList(buf, buffer_IDs, len) == -1 
            && buf->producer_done == 0 ) {
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }

    fullBuf = private_getFullBufferFromList(buf, buffer_IDs, len);

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    return fullBuf;
}

void markProducerDone(struct Buffer* buf)
{
    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->producer_done = 1;

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumers
    HANDLE_ERROR( pthread_cond_broadcast(&buf->full_cond) );
}

int getNumFullBuffers(struct Buffer* buf)
{
    int numFull = 0;

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    for (int i = 0; i < buf->num_buffers; ++i) {
        if (buf->is_full[i] == 1) {
            numFull++;
        }
    }

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );

    return numFull;
}

void printBufferStatus(struct Buffer* buf)
{

    int is_full[buf->num_buffers];

    HANDLE_ERROR( pthread_mutex_lock(&buf->lock) );

    memcpy(is_full, buf->is_full, buf->num_buffers * sizeof(int));

    HANDLE_ERROR( pthread_mutex_unlock(&buf->lock) );
    
    printf("Buffer Status: ");
    for (int i = 0; i < buf->num_buffers; ++i) {
        if (buf->is_full[i] == 1) {
            printf("X");
        } else {
            printf("_");
        }
    }
    printf("\n");
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

