#include "buffers.h"
#include "errors.h"
#include "error_correction.h"
#include "nt_memset.h"
#ifdef WITH_HSA
#include "hsaBase.h"
#endif

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

int create_buffer(struct Buffer* buf, int num_buf, int len,
                  struct InfoObjectPool * pool, const char * buffer_name)
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
    // NOTE (17/02/02) This may not be needed any more, changed to make aligned
    // len == requested len.  This should be checked in more detail.
    buf->aligned_buffer_size = len;
    //buf->aligned_buffer_size = PAGESIZE_MEM * (ceil((double)len / (double)PAGESIZE_MEM));

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
    CHECK_MEM(buf->data);

    // Create the info array
    buf->info = malloc(num_buf*sizeof(void *));
    CHECK_MEM(buf->info);

    for(int i = 0; i < num_buf; ++i) {
        buf->info[i] = NULL;
    }

    for (int i = 0; i < MAX_PRODUCERS; ++i) {
        buf->producers[i].in_use = 0;
    }
    for (int i = 0; i < MAX_CONSUMERS; ++i) {
        buf->consumers[i].in_use = 0;
    }

    // Create the arrays for marking consumers and producers as done.
    buf->producers_done = malloc(num_buf*sizeof(int *));
    CHECK_MEM(buf->producers_done);
    buf->consumers_done = malloc(num_buf*sizeof(int *));
    CHECK_MEM(buf->consumers_done);

    for (int i = 0; i < num_buf; ++i) {
        buf->producers_done[i] = malloc(MAX_PRODUCERS*sizeof(int));
        buf->consumers_done[i] = malloc(MAX_CONSUMERS*sizeof(int));

        CHECK_MEM(buf->producers_done[i]);
        CHECK_MEM(buf->consumers_done[i]);

        private_reset_producers(buf, i);
        private_reset_consumers(buf, i);
    }

    // Create the zero when done array
    buf->zero_buffer = malloc(num_buf*sizeof(int));
    CHECK_MEM(buf->zero_buffer);
    memset(buf->zero_buffer, 0, num_buf*sizeof(int));

    int err = 0;

    // Create the actual buffers.
    for (int i = 0; i < num_buf; ++i) {

        #ifdef WITH_HSA

        // Is this memory aligned?

        buf->data[i] = hsa_host_malloc(buf->aligned_buffer_size);
        INFO("Using hsa_host_malloc in buffers.c: %p, len: %d", buf->data[i], buf->aligned_buffer_size);

        memset(buf->data[i], 0x88888888, buf->aligned_buffer_size);

        #else
        // Create a page alligned block of memory for the buffer
        err = posix_memalign((void **) &(buf->data[i]), PAGESIZE_MEM, buf->aligned_buffer_size);
        CHECK_MEM(buf->data[i]);
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
        #endif
    }

    return 0;
}

void delete_buffer(struct Buffer* buf)
{
    for (int i = 0; i < buf->num_buffers; ++i) {
        free(buf->data[i]);
        free(buf->producers_done[i]);
        free(buf->consumers_done[i]);
    }

    free(buf->data);

    free(buf->is_full);

    free(buf->info);

    free(buf->producers_done);
    free(buf->consumers_done);

    free(buf->zero_buffer);

    // Free locks and cond vars
    CHECK_ERROR( pthread_mutex_destroy(&buf->lock_info) );
    CHECK_ERROR( pthread_mutex_destroy(&buf->lock) );
    CHECK_ERROR( pthread_cond_destroy(&buf->full_cond) );
    CHECK_ERROR( pthread_cond_destroy(&buf->empty_cond) );
}

void mark_buffer_full(struct Buffer * buf, const char * name, const int ID) {
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    //DEBUG("Buffer %s[%d] being marked full by producer %s\n", buf->buffer_name, ID, name);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    int set_full = 0;

    private_mark_producer_done(buf, name, ID);
    if (private_producers_done(buf, ID) == 1) {
        private_reset_producers(buf, ID);
        buf->is_full[ID] = 1;
        set_full = 1;
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumer
    if (set_full == 1) {
        CHECK_ERROR( pthread_cond_broadcast(&buf->full_cond) );
    }
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

void *private_zero_buffer(void * args) {

    int ID = ((struct zero_buffer_thread_args *)(args))->ID;
    struct Buffer * buf = ((struct zero_buffer_thread_args *)(args))->buf;

    assert (ID >= 0);
    assert (ID <= buf->num_buffers);

    // This zeros everything, but for VDIF we just need to header zeroed.
    //int div_256 = 256*(buf->buffer_size / 256);
    //nt_memset((void *)buf->data[ID], 0x00, div_256);
    //memset((void *)&buf->data[ID][div_256], 0x00, buf->buffer_size - div_256);

    // HACK: Just zero the first two words of the VDIF header
    for (int i = 0; i < buf->buffer_size/1056; ++i) {
        *((uint64_t*)&buf->data[ID][i*1056]) = 0;
    }

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

        buf->is_full[ID] = 0;
        private_reset_consumers(buf, ID);
        buf->zero_buffer[ID] = 0;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    CHECK_ERROR( pthread_cond_broadcast(&buf->empty_cond) );

    free(args);

    int ret = 0;
    pthread_exit(&ret);
}

void zero_buffer(struct Buffer * buf, const int ID) {

    assert (ID >= 0);
    assert (ID <= buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );
    buf->zero_buffer[ID] = 1;
    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
}

void mark_buffer_empty(struct Buffer* buf, const char * consumer_name, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);
    int broadcast = 0;

    // If we've been asked to zero the buffer do it here.
    // This needs to happen out side of the critical section
    // so that we don't block for a long time here.
    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

        private_mark_consumer_done(buf, consumer_name, ID);
        if (private_consumers_done(buf, ID) == 1) {

            if (buf->zero_buffer[ID] == 1) {
                pthread_t zero_t;
                struct zero_buffer_thread_args * zero_args = malloc(sizeof(struct zero_buffer_thread_args));
                zero_args->ID = ID;
                zero_args->buf = buf;

                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                for (int j = 8; j < 12; j++)
                    CPU_SET(j, &cpuset);

                CHECK_ERROR( pthread_create(&zero_t, NULL, &private_zero_buffer, (void *)zero_args) );
                CHECK_ERROR( pthread_setaffinity_np(zero_t, sizeof(cpu_set_t), &cpuset) );
            } else {
                buf->is_full[ID] = 0;
                private_reset_consumers(buf, ID);
                buf->zero_buffer[ID] = 0;
                broadcast = 1;
            }
        }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal producer
    if (broadcast == 1) {
        CHECK_ERROR( pthread_cond_broadcast(&buf->empty_cond) );
    }
}

void wait_for_empty_buffer(struct Buffer* buf, const char * producer_name, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_buffers);

    int print_stat = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    int producer_id = private_get_producer_id(buf, producer_name);

    // If the buffer isn't full, i.e. is_full[ID] == 0, then we never sleep on the cond var.
    // The second condition stops us from using a buffer we've already filled,
    // and forces a wait until that buffer has been marked as empty.
    while (buf->is_full[ID] == 1 || buf->producers_done[ID][producer_id] == 1) {
        DEBUG("wait_for_empty_buffer: %s waiting for empty buffer ID = %d in buffer %s",
              producer_name, ID, buf->buffer_name);
        print_stat = 1;
        pthread_cond_wait(&buf->empty_cond, &buf->lock);
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    if (print_stat == 1)
        print_buffer_status(buf);

}

void register_consumer(struct Buffer * buf, const char *name) {
    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    DEBUG("Registering consumer %s for buffer %s", name, buf->buffer_name);

    if (private_get_consumer_id(buf, name) != -1) {
        ERROR("You cannot register two consumers with the same name!");
        assert(0); // Optional
        CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
        return;
    }

    for (int i = 0; i < MAX_CONSUMERS; ++i) {
        if (buf->consumers[i].in_use == 0) {
            buf->consumers[i].in_use = 1;
            strncpy(buf->consumers[i].name, name, MAX_PROCESS_NAME_LEN);
            CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
            return;
        }
    }

    ERROR("No free slot for consumer, please change buffers.h MAX_CONSUMERS");
    assert(0); // Optional

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
}

void register_producer(struct Buffer * buf, const char *name) {
    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    if (private_get_producer_id(buf, name) != -1) {
        ERROR("You cannot register two consumers with the same name!");
        assert(0); // Optional
        CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
        return;
    }

    for (int i = 0; i < MAX_PRODUCERS; ++i) {
        if (buf->producers[i].in_use == 0) {
            buf->producers[i].in_use = 1;
            strncpy(buf->producers[i].name, name, MAX_PROCESS_NAME_LEN);
            CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
            return;
        }
    }

    ERROR("No free slot for producer, please change buffers.h MAX_PRODUCERS");
    assert(0); // Optional

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
}

int private_get_consumer_id(struct Buffer * buf, const char * name) {

    for (int i = 0; i < MAX_CONSUMERS; ++i) {
        if (buf->consumers[i].in_use == 1 &&
            strncmp(buf->consumers[i].name, name, MAX_PROCESS_NAME_LEN) == 0) {
            return i;
        }
    }
    return -1;
}

int private_get_producer_id(struct Buffer * buf, const char * name) {

    for (int i = 0; i < MAX_PRODUCERS; ++i) {
        if (buf->producers[i].in_use == 1 &&
            strncmp(buf->producers[i].name, name, MAX_PROCESS_NAME_LEN) == 0) {
            return i;
        }
    }
    return -1;
}

void private_reset_producers(struct Buffer * buf, const int ID) {
    memset(buf->producers_done[ID], 0, MAX_PRODUCERS*sizeof(int));
}

void private_reset_consumers(struct Buffer * buf, const int ID) {
    memset(buf->consumers_done[ID], 0, MAX_CONSUMERS*sizeof(int));
}

void private_mark_consumer_done(struct Buffer * buf, const char * name, const int ID) {
    int consumer_id = private_get_consumer_id(buf, name);
    if (consumer_id == -1) {
        ERROR("The consumer %s hasn't been registered!", name);
    }

    //DEBUG("%s->consumers_done[%d][%d] == %d", buf->buffer_name, ID, consumer_id, buf->consumers_done[ID][consumer_id] );

    assert(consumer_id != -1);
    // The consumer we are marking as done, shouldn't already be done!
    assert(buf->consumers_done[ID][consumer_id] == 0);

    buf->consumers_done[ID][consumer_id] = 1;
}

void private_mark_producer_done(struct Buffer * buf, const char * name, const int ID) {
    int producer_id = private_get_producer_id(buf, name);
    if (producer_id == -1) {
        ERROR("The producer %s hasn't been registered!", name);
    }

    //DEBUG("%s->producers_done[%d][%d] == %d", buf->buffer_name, ID, producer_id, buf->producers_done[ID][producer_id] );

    assert(producer_id != -1);
    // The producer we are marking as done, shouldn't already be done!
    assert(buf->producers_done[ID][producer_id] == 0);

    buf->producers_done[ID][producer_id] = 1;
}

int private_consumers_done(struct Buffer * buf, const int ID) {

    for (int i = 0; i < MAX_CONSUMERS; ++i) {
        if (buf->consumers[i].in_use == 1 && buf->consumers_done[ID][i] == 0)
            return 0;
    }
    return 1;
}

int private_producers_done(struct Buffer * buf, const int ID) {

    for (int i = 0; i < MAX_CONSUMERS; ++i) {
        if (buf->producers[i].in_use == 1 && buf->producers_done[ID][i] == 0)
            return 0;
    }
    return 1;
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

uint64_t get_fpga_seq_num(struct Buffer* buf, const int ID)
{
    uint64_t fpga_seq_num = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    if (buf->info[ID] == NULL) {
        WARN("get_fpga_seq_num: info struct %d is null", ID);
    }
    assert(buf->info[ID] != NULL);
    fpga_seq_num = buf->info[ID]->fpga_seq_num;

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock_info) );

    return fpga_seq_num;
}

int32_t get_streamID(struct Buffer* buf, const int ID)
{
    int stream_ID = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock_info) );

    if (buf->info[ID] == NULL) {
        WARN("get_streamID: info struct %d is null", ID);
        stream_ID = -1;
    } else {
        stream_ID = buf->info[ID]->stream_ID;
    }

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

void set_fpga_seq_num(struct Buffer* buf, const int ID, const uint64_t fpga_seq_num)
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


int wait_for_full_buffer(struct Buffer* buf, const char * name, const int ID)
{
    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    int consumer_id = private_get_consumer_id(buf, name);

    // This loop exists when is_full == 1 (i.e. a full buffer) AND
    // when this producer hasn't already marked this buffer as
    while ( buf->is_full[ID] == 0 ||
            buf->consumers_done[ID][consumer_id] == 1 ) {
        pthread_cond_wait(&buf->full_cond, &buf->lock);
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // TODO Enable -1 return when no producers exist.
    return ID;
}

void mark_producer_done(struct Buffer* buf, int producer_id)
{
    assert(buf != NULL);
    assert(producer_id >= 0);
    assert(producer_id < buf->num_buffers);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    //buf->producer_done[producer_id] = 1;
    INFO("mark_producer_done() doesn't do anything for now...");

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

    char status_string[buf->num_buffers + 1];

    for (int i = 0; i < buf->num_buffers; ++i) {
        if (buf->is_full[i] == 1) {
            status_string[i] = 'X';
        } else {
            status_string[i] = '_';
        }
    }
    status_string[buf->num_buffers] = '\0';
    DEBUG("Buffer %s status: %s", buf->buffer_name, status_string);
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

void copy_buffer_info(struct Buffer * from, int from_id, struct Buffer * to, int to_id)
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
    to->info[to_id]->ref_count++;

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

    ret->ref_count = 1;

    CHECK_ERROR( pthread_mutex_unlock(&pool->in_use_lock) );

    return ret;
}

void return_info_object(struct InfoObjectPool * pool, struct BufferInfo * buffer_info)
{
    CHECK_ERROR( pthread_mutex_lock(&pool->in_use_lock) );

    if (--buffer_info->ref_count == 0) {
        reset_info_object(buffer_info);
        buffer_info->in_use = 0;
    }

    CHECK_ERROR( pthread_mutex_unlock(&pool->in_use_lock) );
}

void reset_info_object(struct BufferInfo * buffer_info)
{
    buffer_info->data_ID = -1;
    reset_error_matrix(&buffer_info->error_matrix);
    buffer_info->fpga_seq_num = 0;
    buffer_info->ref_count = 0;
}

void delete_info_object_pool(struct InfoObjectPool * pool)
{
    for (int i = 0; i < pool->pool_size; ++i) {
        delete_error_matrix( &pool->info_objects[i].error_matrix );
    }
    free(pool->info_objects);
    pthread_mutex_destroy(&pool->in_use_lock);
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
    //int result = 1;
    //for (int i = 0; i < buf->num_producers; ++i) {
    //    if (buf->producer_done[i] == 0) {
    //        result = 0;
    //        break;
    //    }
    //}
    return 0;
    //return result;
}


