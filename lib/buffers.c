#include "buffers.h"
#include "errors.h"
#include "error_correction.h"

#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <math.h>


/**
 * @brief private function to get the id of the first full buffer.
 * This should only be called from within a lock.
 */
int private_get_full_buffer(struct Buffer * buf);

/**
 * @brief finds a full buffer within a list of buffers, and returns its ID if it exists.
 * This should only be called from within a lock.
 */
int private_get_full_buffer_from_list(struct Buffer * buf, const int* buffer_IDs, const int len);

int private_are_producers_done(struct Buffer * buf);

// Checks if the buffer in question has an BufferInfo object attached to it.
// If not it requests one from the buffer_info pool.
// Not thread safe, call from with-in a lock.
void private_check_info_object(struct Buffer * buf, const int ID);

int create_buffer(struct Buffer* buf, int num_buf, int len, int num_producers,
                  struct InfoObjectPool * pool, char * buffer_name)
{

    assert(num_buf > 0);
    assert(pool != NULL);

    CHECK_ERROR( pthread_mutex_init(&buf->lock, NULL) );
    CHECK_ERROR( pthread_mutex_init(&buf->lock_info, NULL) );

    CHECK_ERROR( pthread_cond_init(&buf->full_cond, NULL) );
    CHECK_ERROR( pthread_cond_init(&buf->empty_cond, NULL) );

    // Copy the buffer buffer name.
    buf->buffer_name = strdup(buffer_name);

    buf->num_buffers = num_buf;
    buf->info_object_pool = pool;
    buf->buffer_size = len;
    // We align the buffer length to a multiple of the system page size.
    // This may result in the memory allocated being larger than the size of the
    // memory requested.  So buffer_size is the size requested/used, and aligned_buffer_size
    // is the actual size of the memory space.
    // To make CPU-GPU transfers more efficient, it is recommended to use the aligned value
    // so that no partial pages are send in the DMA copy.
    buf->aligned_buffer_size = PAGESIZE_MEM * (ceil((double)len / (double)PAGESIZE_MEM));
    buf->num_producers = num_producers;

    // Make sure we don't have a math error,
    // which would make the buffer smaller than it should be.
    assert(buf->aligned_buffer_size >= buf->buffer_size);

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
    buf->info = malloc(num_buf*sizeof(void *));

    if ( buf->info == NULL ) {
        ERROR("Error creating info array");
        return errno;
    }

    for(int i = 0; i < num_buf; ++i) {
        buf->info[i] = NULL;
    }

    // Create the array for tracking producers
    buf->producer_done = malloc(num_producers * sizeof(int));
    CHECK_MEM(buf->producer_done);

    memset(buf->producer_done, 0, num_producers * sizeof(int));

    int err = 0;

    // Create the actual buffers.
    for (int i = 0; i < num_buf; ++i) {

        // Create a page alligned block of memory for the buffer
        err = posix_memalign((void **) &(buf->data[i]), PAGESIZE_MEM, buf->aligned_buffer_size);

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

void delete_buffer(struct Buffer* buf)
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

void mark_buffer_full(struct Buffer * buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->is_full[ID] = 1;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumer
    CHECK_ERROR( pthread_cond_broadcast(&buf->full_cond) );
}

int get_full_buffer_ID(struct Buffer * buf)
{
    int fullBuf = -1;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    while ( private_get_full_buffer(buf) == -1
            && private_are_producers_done(buf) == 0) {
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }

    fullBuf = private_get_full_buffer(buf);

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    return fullBuf;
}

void mark_buffer_empty(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    buf->is_full[ID] = 0;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal producer
    CHECK_ERROR( pthread_cond_broadcast(&buf->empty_cond) );
}

void wait_for_empty_buffer(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    // If the buffer isn't full, i.e. is_full[ID] == 0, then we never sleep on the cond var.
    while (buf->is_full[ID] == 1) {
        DEBUG("wait_for_empty_buffer: waiting for empty buffer ID = %d in buffer %s",
              ID, buf->buffer_name);
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

    assert(buf->info[ID] != NULL);
    dataID = buf->info[ID]->data_ID;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return dataID;
}

uint32_t get_fpga_seq_num(struct Buffer* buf, const int ID)
{
    int fpga_seq_num = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    if (buf->info[ID] == NULL) {
        WARN("get_fpga_seq_num: info struct %d is null", ID);
    }
    assert(buf->info[ID] != NULL);
    fpga_seq_num = buf->info[ID]->fpga_seq_num;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return fpga_seq_num;
}

uint16_t get_streamID(struct Buffer* buf, const int ID)
{
    int stream_ID = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    if (buf->info[ID] == NULL) {
        WARN("get_streamID: info struct %d is null", ID);
    }
    assert(buf->info[ID] != NULL);
    stream_ID = buf->info[ID]->stream_ID;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return stream_ID;
}

struct timeval get_first_packet_recv_time(struct Buffer* buf, const int ID)
{
    struct timeval time;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    assert(buf->info[ID] != NULL);
    time = buf->info[ID]->first_packet_recv_time;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return time;
}

struct ErrorMatrix * get_error_matrix(struct Buffer * buf, const int ID)
{
    struct ErrorMatrix * ret = NULL;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    private_check_info_object(buf, ID);
    ret = &buf->info[ID]->error_matrix;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    // TODO By operating on the error matrix like this we break thread safety
    // of the BufferInfo sturct.  See comment on decl.
    return ret;
}

void set_data_ID(struct Buffer* buf, const int ID, const int data_ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    private_check_info_object(buf, ID);
    buf->info[ID]->data_ID = data_ID;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}

void set_fpga_seq_num(struct Buffer* buf, const int ID, const uint32_t fpga_seq_num)
{
    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    private_check_info_object(buf, ID);
    buf->info[ID]->fpga_seq_num = fpga_seq_num;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}

void set_stream_ID(struct Buffer* buf, const int ID, const uint16_t stream_ID)
{
    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    private_check_info_object(buf, ID);
    buf->info[ID]->stream_ID = stream_ID;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) )
}


void set_first_packet_recv_time(struct Buffer* buf, const int ID, struct timeval time)
{
    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    private_check_info_object(buf, ID);
    buf->info[ID]->first_packet_recv_time = time;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}


int get_full_buffer_from_list(struct Buffer* buf, const int* buffer_IDs, const int len)
{
    int fullBuf = -1;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    while ( private_get_full_buffer_from_list(buf, buffer_IDs, len) == -1
            && private_are_producers_done(buf) == 0 ) {
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }

    fullBuf = private_get_full_buffer_from_list(buf, buffer_IDs, len);
    //buf->is_full[fullBuf] = 2;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    return fullBuf;
}

void mark_producer_done(struct Buffer* buf, int producer_id)
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

int get_num_full_buffers(struct Buffer* buf)
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

void print_buffer_status(struct Buffer* buf)
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

void move_buffer_info(struct Buffer * from, int from_id, struct Buffer * to, int to_id)
{
    assert(from != NULL);
    assert(to != NULL);

    CHECK_ERROR( pthread_mutex_lock(&from->lock_info) );
    CHECK_ERROR( pthread_mutex_lock(&to->lock_info) );

    // Assume we are always copying an already valid pointer.
    assert(from->info[from_id] != NULL);

    // Assume we are always coping to a buffer without a valid pointer.
    assert(to->info[to_id] == NULL);

    to->info[to_id] = from->info[from_id];
    // Only one buffer object should ever have control of a BufferInfo object.
    from->info[from_id] = NULL;
    //INFO("move_buffer_info: Setting info id: %d to null", from_id);

    CHECK_ERROR( pthread_mutex_unlock(&to->lock_info) );
    CHECK_ERROR( pthread_mutex_unlock(&from->lock_info) );
}

void release_info_object(struct Buffer * buf, const int ID)
{
    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    return_info_object(buf->info_object_pool, buf->info[ID]);
    buf->info[ID] = NULL;
    //INFO("release_info_object: Setting info id: %d to null", ID);

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );
}

void private_check_info_object(struct Buffer * buf, const int ID)
{
    assert(buf != NULL);
    assert(buf->info != NULL);
    assert(ID >= 0);
    assert(ID < buf->num_buffers);

    if (buf->info[ID] == NULL) {
        buf->info[ID] = request_info_object(buf->info_object_pool);
    }

    // We assume for now that we always have enough info objects in the pool.
    assert(buf->info[ID] != NULL);
}

void create_info_pool(struct InfoObjectPool * pool, int num_info_objects, int num_freq, int num_elem)
{
    CHECK_ERROR( pthread_mutex_init(&pool->in_use_lock, NULL) );

    pool->info_objects = malloc(num_info_objects * sizeof(struct BufferInfo));
    CHECK_MEM(pool->info_objects);

    pool->pool_size = num_info_objects;

    for (int i = 0; i < num_info_objects; ++i) {
        initalize_error_matrix(&pool->info_objects[i].error_matrix, num_freq, num_elem);
        reset_info_object(&pool->info_objects[i]);
        pool->info_objects[i].in_use = 0;
    }

    CHECK_ERROR( pthread_mutex_unlock(&pool->in_use_lock) );
}

struct BufferInfo * request_info_object(struct InfoObjectPool * pool) {

    struct BufferInfo * ret = NULL;

    CHECK_ERROR( pthread_mutex_lock(&pool->in_use_lock) );

    for (int i = 0; i < pool->pool_size; ++i) {
        if (pool->info_objects[i].in_use == 0) {
            pool->info_objects[i].in_use = 1;
            ret = &pool->info_objects[i];
            break;
        }
    }

    // Assume we never run out.
    assert(ret != NULL);

    CHECK_ERROR( pthread_mutex_unlock(&pool->in_use_lock) );

    return ret;
}

void return_info_object(struct InfoObjectPool * pool, struct BufferInfo * buffer_info)
{
    CHECK_ERROR( pthread_mutex_lock(&pool->in_use_lock) );

    reset_info_object(buffer_info);
    buffer_info->in_use = 0;

    CHECK_ERROR( pthread_mutex_unlock(&pool->in_use_lock) );
}

void reset_info_object(struct BufferInfo * buffer_info)
{
    buffer_info->data_ID = -1;
    reset_error_matrix(&buffer_info->error_matrix);
    buffer_info->fpga_seq_num = 0;
}

void delete_info_object_pool(struct InfoObjectPool * pool)
{
    for (int i = 0; i < pool->pool_size; ++i) {
        delete_error_matrix( &pool->info_objects[i].error_matrix );
    }
    CHECK_ERROR( pthread_mutex_destroy(&pool->in_use_lock) );
}

int private_get_full_buffer(struct Buffer* buf)
{
    for (int i = 0; i < buf->num_buffers; ++i) {
        if (buf->is_full[i] == 1) {
            return i;
        }
    }
    return -1;
}

int private_get_full_buffer_from_list(struct Buffer* buf, const int* buffer_IDs, const int len)
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

int private_are_producers_done(struct Buffer* buf)
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


