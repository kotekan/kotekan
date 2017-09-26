#include "buffer.h"
#include "metadata.h"
#include "errors.h"
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


struct zero_frames_thread_args {
    struct Buffer * buf;
    int ID;
};

void *private_zero_frames(void * args);

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


int create_buffer(struct Buffer* buf, int num_frames, int len,
                  struct metadataPool * pool, const char * buffer_name)
{

    assert(num_frames > 0);
    assert(pool != NULL);

    CHECK_ERROR( pthread_mutex_init(&buf->lock, NULL) );

    CHECK_ERROR( pthread_cond_init(&buf->full_cond, NULL) );
    CHECK_ERROR( pthread_cond_init(&buf->empty_cond, NULL) );

    // Copy the buffer buffer name.
    buf->buffer_name = strdup(buffer_name);

    buf->num_frames = num_frames;
    buf->metadata_pool = pool;
    buf->frame_size = len;

    // We align the buffer length to a multiple of the system page size.
    // This may result in the memory allocated being larger than the size of the
    // memory requested.  So frame_size is the size requested/used, and aligned_frame_size
    // is the actual size of the memory space.
    // To make CPU-GPU transfers more efficient, it is recommended to use the aligned value
    // so that no partial pages are send in the DMA copy.
    // NOTE (17/02/02) This may not be needed any more, changed to make aligned
    // len == requested len.  This should be checked in more detail.
    buf->aligned_frame_size = len;
    //buf->aligned_frame_size = PAGESIZE_MEM * (ceil((double)len / (double)PAGESIZE_MEM));

    // Make sure we don't have a math error,
    // which would make the buffer smaller than it should be.
    assert(buf->aligned_frame_size >= buf->frame_size);

    // Create the is_free array
    buf->is_full = malloc(num_frames * sizeof(int));

    if ( buf->is_full == NULL ) {
        ERROR("Error creating is_full array");
        return errno;
    }

    memset(buf->is_full, 0, num_frames*sizeof(int));

    // Create the array of buffer pointers.
    buf->frames = malloc(num_frames * sizeof(void *));
    CHECK_MEM(buf->frames);

    // Create the info array
    buf->metadata = malloc(num_frames*sizeof(void *));
    CHECK_MEM(buf->metadata);

    for(int i = 0; i < num_frames; ++i) {
        buf->metadata[i] = NULL;
    }

    for (int i = 0; i < MAX_PRODUCERS; ++i) {
        buf->producers[i].in_use = 0;
    }
    for (int i = 0; i < MAX_CONSUMERS; ++i) {
        buf->consumers[i].in_use = 0;
    }

    // Create the arrays for marking consumers and producers as done.
    buf->producers_done = malloc(num_frames*sizeof(int *));
    CHECK_MEM(buf->producers_done);
    buf->consumers_done = malloc(num_frames*sizeof(int *));
    CHECK_MEM(buf->consumers_done);

    for (int i = 0; i < num_frames; ++i) {
        buf->producers_done[i] = malloc(MAX_PRODUCERS*sizeof(int));
        buf->consumers_done[i] = malloc(MAX_CONSUMERS*sizeof(int));

        CHECK_MEM(buf->producers_done[i]);
        CHECK_MEM(buf->consumers_done[i]);

        private_reset_producers(buf, i);
        private_reset_consumers(buf, i);
    }

    // By default don't zero buffers at the end of their use.
    buf->zero_frames = 0;

    // Create the frames.
    for (int i = 0; i < num_frames; ++i) {

        #ifdef WITH_HSA

        // Is this memory aligned?
        buf->frames[i] = hsa_host_malloc(buf->aligned_frame_size);
        INFO("Using hsa_host_malloc in buffers.c: %p, len: %d", buf->frames[i], buf->aligned_frame_size);

        memset(buf->frames[i], 0x88888888, buf->aligned_frame_size);

        #else
        // Create a page alligned block of memory for the buffer
        int err = 0;
        err = posix_memalign((void **) &(buf->frames[i]), PAGESIZE_MEM, buf->aligned_frame_size);
        CHECK_MEM(buf->frames[i]);
        if ( err != 0 ) {
            ERROR("Error creating alligned memory");
            return err;
        }

        // Ask that all pages be kept in memory
        err = mlock((void *) buf->frames[i], len);

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
    for (int i = 0; i < buf->num_frames; ++i) {
        free(buf->frames[i]);
        free(buf->producers_done[i]);
        free(buf->consumers_done[i]);
    }

    free(buf->frames);
    free(buf->is_full);
    free(buf->metadata);
    free(buf->producers_done);
    free(buf->consumers_done);

    // Free locks and cond vars
    CHECK_ERROR( pthread_mutex_destroy(&buf->lock) );
    CHECK_ERROR( pthread_cond_destroy(&buf->full_cond) );
    CHECK_ERROR( pthread_cond_destroy(&buf->empty_cond) );
}

void mark_frame_full(struct Buffer * buf, const char * name, const int ID) {
    assert (ID >= 0);
    assert (ID < buf->num_frames);

    //DEBUG("Frame %s[%d] being marked full by producer %s\n", buf->buffer_name, ID, name);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    int set_full = 0;

    private_mark_producer_done(buf, name, ID);
    if (private_producers_done(buf, ID) == 1) {
        private_reset_producers(buf, ID);
        buf->is_full[ID] = 1;
        set_full = 1;
    }

    // If there are no consumers registered then we can just mark the buffer empty
    if (private_consumers_done(buf, ID) == 1) {
        INFO("No consumers are registered on %s dropping data in frame %d...", buf->buffer_name, ID);
        buf->is_full[ID] = 0;
        private_reset_consumers(buf, ID);
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal consumer
    if (set_full == 1) {
        CHECK_ERROR( pthread_cond_broadcast(&buf->full_cond) );
    }
}

void *private_zero_frames(void * args) {

    int ID = ((struct zero_frames_thread_args *)(args))->ID;
    struct Buffer * buf = ((struct zero_frames_thread_args *)(args))->buf;

    assert (ID >= 0);
    assert (ID <= buf->num_frames);

    // This zeros everything, but for VDIF we just need to header zeroed.
    //int div_256 = 256*(buf->frame_size / 256);
    //nt_memset((void *)buf->frames[ID], 0x00, div_256);
    //memset((void *)&buf->frames[ID][div_256], 0x00, buf->frame_size - div_256);

    // HACK: Just zero the first two words of the VDIF header
    for (int i = 0; i < buf->frame_size/1056; ++i) {
        *((uint64_t*)&buf->frames[ID][i*1056]) = 0;
    }

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

        buf->is_full[ID] = 0;
        private_reset_consumers(buf, ID);

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    CHECK_ERROR( pthread_cond_broadcast(&buf->empty_cond) );

    free(args);

    int ret = 0;
    pthread_exit(&ret);
}

void zero_frames(struct Buffer * buf) {
    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );
    buf->zero_frames = 1;
    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
}

void mark_frame_empty(struct Buffer* buf, const char * consumer_name, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_frames);
    int broadcast = 0;

    // If we've been asked to zero the buffer do it here.
    // This needs to happen out side of the critical section
    // so that we don't block for a long time here.
    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

        private_mark_consumer_done(buf, consumer_name, ID);
        if (private_consumers_done(buf, ID) == 1) {

            if (buf->zero_frames == 1) {
                pthread_t zero_t;
                struct zero_frames_thread_args * zero_args = malloc(sizeof(struct zero_frames_thread_args));
                zero_args->ID = ID;
                zero_args->buf = buf;

                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                // TODO: Move this to the config file (when buffers.c updated to C++11)
                CPU_SET(5, &cpuset);

                CHECK_ERROR( pthread_create(&zero_t, NULL, &private_zero_frames, (void *)zero_args) );
                CHECK_ERROR( pthread_setaffinity_np(zero_t, sizeof(cpu_set_t), &cpuset) );
                CHECK_ERROR( pthread_detach(zero_t) );
            } else {
                buf->is_full[ID] = 0;
                private_reset_consumers(buf, ID);
                broadcast = 1;
            }
            decrement_metadata_ref_count(buf->metadata[ID]);
            buf->metadata[ID] = NULL;
        }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    // Signal producer
    if (broadcast == 1) {
        CHECK_ERROR( pthread_cond_broadcast(&buf->empty_cond) );
    }
}

uint8_t * wait_for_empty_frame(struct Buffer* buf, const char * producer_name, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_frames);

    int print_stat = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    int producer_id = private_get_producer_id(buf, producer_name);

    // If the buffer isn't full, i.e. is_full[ID] == 0, then we never sleep on the cond var.
    // The second condition stops us from using a buffer we've already filled,
    // and forces a wait until that buffer has been marked as empty.
    while (buf->is_full[ID] == 1 || buf->producers_done[ID][producer_id] == 1) {
        DEBUG("wait_for_empty_frame: %s waiting for empty frame ID = %d in buffer %s",
              producer_name, ID, buf->buffer_name);
        print_stat = 1;
        pthread_cond_wait(&buf->empty_cond, &buf->lock);
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    if (print_stat == 1)
        print_buffer_status(buf);

    return buf->frames[ID];
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

    ERROR("No free slot for consumer, please change buffer.h MAX_CONSUMERS");
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

    ERROR("No free slot for producer, please change buffer.h MAX_PRODUCERS");
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

int is_frame_empty(struct Buffer* buf, const int ID)
{
    assert (ID >= 0);
    assert (ID < buf->num_frames);
    assert (buf != NULL);

    int empty = 1;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    if (buf->is_full[ID] == 1) {
        empty = 0;
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    return empty;
}

uint8_t * wait_for_full_frame(struct Buffer* buf, const char * name, const int ID)
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
    return buf->frames[ID];
}

int get_num_full_frames(struct Buffer* buf)
{
    int numFull = 0;

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    for (int i = 0; i < buf->num_frames; ++i) {
        if (buf->is_full[i] == 1) {
            numFull++;
        }
    }

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    return numFull;
}

void print_buffer_status(struct Buffer* buf)
{
    int is_full[buf->num_frames];

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    memcpy(is_full, buf->is_full, buf->num_frames * sizeof(int));

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );

    char status_string[buf->num_frames + 1];

    for (int i = 0; i < buf->num_frames; ++i) {
        if (buf->is_full[i] == 1) {
            status_string[i] = 'X';
        } else {
            status_string[i] = '_';
        }
    }
    status_string[buf->num_frames] = '\0';
    DEBUG("Buffer %s status: %s", buf->buffer_name, status_string);
}

void pass_metadata(struct Buffer * from_buf, int from_ID, struct Buffer * to_buf, int to_ID) {

    struct metadataContainer * metadata_container = NULL;

    assert(from_buf->metadata[from_ID] != NULL);
    metadata_container = from_buf->metadata[from_ID];

    CHECK_ERROR( pthread_mutex_lock(&to_buf->lock) );

    // In case we've already moved the metadata we don't want to increment the ref count.
    if (to_buf->metadata[to_ID] == NULL) {
        to_buf->metadata[to_ID] = metadata_container;
        increment_metadata_ref_count(metadata_container);
    }

    // If this is true then the to_buf already has a metadata container for this ID and its different!
    assert(to_buf->metadata[to_ID] == metadata_container);
    CHECK_ERROR( pthread_mutex_unlock(&to_buf->lock) );
}

void allocate_new_metadata_object(struct Buffer * buf, int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);

    CHECK_ERROR( pthread_mutex_lock(&buf->lock) );

    INFO("Called allocate_new_metadata_object, buf %p, %d", buf, ID);

    if (buf->metadata[ID] == NULL) {
        buf->metadata[ID] = request_metadata_object(buf->metadata_pool);
    }

    // We assume for now that we always have enough info objects in the pool.
    assert(buf->metadata[ID] != NULL);

    CHECK_ERROR( pthread_mutex_unlock(&buf->lock) );
}

void * get_metadata(struct Buffer * buf, int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);
    assert(buf->metadata[ID] != NULL);

    return buf->metadata[ID]->metadata;
}

struct metadataContainer * get_metadata_container(struct Buffer * buf, int ID) {
    assert(ID >= 0);
    assert(ID < buf->num_frames);
    assert(buf->metadata[ID] != NULL);

    return buf->metadata[ID];
}