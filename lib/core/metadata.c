#include "metadata.h"

#include "errors.h" // for CHECK_ERROR_F, CHECK_MEM_F

#include <assert.h> // for assert
#include <stdlib.h> // for malloc, free
#include <string.h> // for memset

// *** Metadata object section ***

struct metadataContainer* create_metadata(size_t object_size, struct metadataPool* parent_pool) {

    struct metadataContainer* metadata_container;
    metadata_container = malloc(sizeof(struct metadataContainer));
    CHECK_MEM_F(metadata_container);

    metadata_container->metadata = malloc(object_size);
    CHECK_MEM_F(metadata_container->metadata);
    metadata_container->metadata_size = object_size;

    metadata_container->ref_count = 0;
    metadata_container->parent_pool = parent_pool;

    reset_metadata_object(metadata_container);

    CHECK_ERROR_F(pthread_mutex_init(&metadata_container->metadata_lock, NULL));

    return metadata_container;
}

void delete_metadata(struct metadataContainer* container) {
    // We might shutdown the system without actually flushing all the buffer
    // chains, so we don't need to assert that the ref_count be 0,
    // at least for now...
    // assert(container->ref_count == 0);
    free(container->metadata);
    CHECK_ERROR_F(pthread_mutex_destroy(&container->metadata_lock));
    free(container);
}

void reset_metadata_object(struct metadataContainer* container) {
    assert(container->ref_count == 0);
    memset(container->metadata, 0, container->metadata_size);
}

void lock_metadata(struct metadataContainer* container) {
    CHECK_ERROR_F(pthread_mutex_lock(&container->metadata_lock));
}

void unlock_metadata(struct metadataContainer* container) {
    CHECK_ERROR_F(pthread_mutex_unlock(&container->metadata_lock));
}

void increment_metadata_ref_count(struct metadataContainer* container) {
    lock_metadata(container);
    container->ref_count += 1;
    unlock_metadata(container);
}

void decrement_metadata_ref_count(struct metadataContainer* container) {
    uint32_t local_ref_count;

    lock_metadata(container);

    assert(container->ref_count > 0);

    container->ref_count -= 1;
    local_ref_count = container->ref_count;

    unlock_metadata(container);

    if (local_ref_count == 0) {
        return_metadata_to_pool(container->parent_pool, container);
    }
}

// *** Metadata pool section ***

struct metadataPool* create_metadata_pool(int num_metadata_objects, size_t object_size) {
    struct metadataPool* pool;
    pool = malloc(sizeof(struct metadataPool));
    CHECK_MEM_F(pool);

    pool->pool_size = num_metadata_objects;
    pool->metadata_object_size = object_size;
    CHECK_ERROR_F(pthread_mutex_init(&pool->pool_lock, NULL));

    pool->in_use = malloc(pool->pool_size * sizeof(int));
    CHECK_MEM_F(pool->in_use);
    pool->metadata_objects = malloc(pool->pool_size * sizeof(struct metadataContainer*));
    CHECK_MEM_F(pool->metadata_objects);

    for (unsigned int i = 0; i < pool->pool_size; ++i) {
        pool->metadata_objects[i] = create_metadata(object_size, pool);
        pool->in_use[i] = 0;
    }

    return pool;
}

void delete_metadata_pool(struct metadataPool* pool) {
    for (unsigned int i = 0; i < pool->pool_size; ++i) {
        delete_metadata(pool->metadata_objects[i]);
    }

    CHECK_ERROR_F(pthread_mutex_destroy(&pool->pool_lock));
    free(pool->metadata_objects);
    free(pool->in_use);
}

struct metadataContainer* request_metadata_object(struct metadataPool* pool) {
    struct metadataContainer* container = NULL;

    CHECK_ERROR_F(pthread_mutex_lock(&pool->pool_lock));

    // TODO there are better data structures for this.
    for (unsigned int i = 0; i < pool->pool_size; ++i) {
        if (pool->in_use[i] == 0) {
            // DEBUG_F("pool->metadata_objects[%d] == %p", i, pool->metadata_objects[i]);
            container = pool->metadata_objects[i];
            assert(container->ref_count == 0); // Shouldn't give an inuse object (!)
            container->ref_count = 1;
            pool->in_use[i] = 1;
            break;
        }
    }
    CHECK_ERROR_F(pthread_mutex_unlock(&pool->pool_lock));

    // We will assume that we cannot use more containers than are in the pool.
    // If you hit this increase your pool size.
    assert(container != NULL);

    return container;
}

void return_metadata_to_pool(struct metadataPool* pool, struct metadataContainer* info) {

    CHECK_ERROR_F(pthread_mutex_lock(&pool->pool_lock));

    // DEBUG_F("Called return_metadata_to_pool");
    for (unsigned int i = 0; i < pool->pool_size; ++i) {
        // Pointer check
        if (pool->metadata_objects[i] == info) {
            assert(pool->in_use[i] == 1); // Should be in-use if we are returning it!

            reset_metadata_object(pool->metadata_objects[i]);
            pool->in_use[i] = 0;
            CHECK_ERROR_F(pthread_mutex_unlock(&pool->pool_lock));
            return;
        }
    }
    assert(0 == 1); // We should have found the info object in the pool (!)
}
