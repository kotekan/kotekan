/**
 * @file
 * @brief The kotekan buffer metadata containers and pools.
 * Most of these functions are used by buffer.c internally and not
 * intended for use outside of the buffer content.
 * - metadataContainer
 * -- create_metadata
 * -- delete_metadata
 * -- reset_metadata_object
 * -- increment_metadata_ref_count
 * -- decrement_metadata_ref_count
 * -- lock_metadata
 * -- unlock_metadata
 * - metadataPool
 * -- create_metadata_pool
 * -- delete_metadata_pool
 * -- request_metadata_object
 * -- return_metadata_to_pool
 */

#ifndef METADATA_H
#define METADATA_H

#include <pthread.h> // for pthread_mutex_t
#include <stdint.h>  // for uint32_t
#include <stdio.h>   // for size_t

struct metadataPool;

#ifdef __cplusplus
extern "C" {
#endif

// *** Metadata object section ***

/**
 * @struct metadataContainer
 * @brief Container which holds the pointer to the actual metadata values
 *
 * The struct is used mostly to hold the reference count and lock variables,
 * as well as the pointer to the actual metadata memory and it's expected size.
 *
 * This container always belongs to a pool of metadata containers,
 * see @c metadataPool for more informaiton
 *
 * @author Andre Renard
 */
struct metadataContainer {

    /// Pointer to the memory where the actual metadata is stored
    void* metadata;
    /// The size of the metadata in bytes
    size_t metadata_size;

    /**
     * @brief Pointer reference count.
     * Tracks references to this object,
     * and returns the object to the associated @c metadataPool, once
     * the counter reaches zero.
     */
    uint32_t ref_count;

    /**
     * @brief Lock for variables in this object.
     * Can also be used to lock access to metadata values.
     */
    pthread_mutex_t metadata_lock;

    /// Reference to metadataPool that this object belongs too.
    struct metadataPool* parent_pool;
};

/**
 * @brief Creates a metadata container with a metadata memory block of the requested size.
 *
 * This function is normally just used by the @c create_metadata_pool to create
 * the container for a given metadata type
 *
 * Note there isn't an explicit type here, just the size of the struct used
 * to contain whatever type of metadata the user wants.
 *
 * @param[in] object_size The size of the metadata struct to be stored in this container
 * @param[in] parent_pool The pool this container will belong too.
 * @return A @c metadataContainer object with the @c metadata memory allocated
 */
struct metadataContainer* create_metadata(size_t object_size, struct metadataPool* parent_pool);

/**
 * @brief Frees the memory associated with a metadataContainer
 * @param[in] container The @c metadataContainer to free
 */
void delete_metadata(struct metadataContainer* container);

/**
 * @brief Zeros the metadata memory region.
 *
 * Used to make sure the metadata starts out as zero when a new container is requested.
 *
 * @param container The container to zero memory for.
 */
void reset_metadata_object(struct metadataContainer* container);

/**
 * @brief Increments the metadata ref counter
 * @param[in] container The container to increment the reference counter for.
 */
void increment_metadata_ref_count(struct metadataContainer* container);

/**
 * @brief Decrements the metadata ref counter
 * @param[in] container The container to decrement the reference counter for.
 */
void decrement_metadata_ref_count(struct metadataContainer* container);

/**
 * @brief Request the lock on the metadata container
 *
 * Used for example when changing the reference counter
 *
 * @param[in] container The container to request the lock for
 */
void lock_metadata(struct metadataContainer* container);

/**
 * @brief Unlocks the lock associated with the metadata container
 * @param[in] container The container to unlock
 */
void unlock_metadata(struct metadataContainer* container);

// *** Metadata pool section ***

/**
 * @brief A memory pool for preallocated metadata containers.
 *
 * The idea behind metadata containers is to be able to pass metadata down
 * a pipeline chain between buffers and stages without copying any of the values
 * at each step, or allocating and deallocating memory.
 *
 * This pool is what holds the metadata containers when they aren't in use
 * and provides references to containers once they are requested.
 *
 * When the a metadata container's reference counter reaches zero, it returns
 * itself back to its associated pool
 *
 * @author Andre Renard
 */
struct metadataPool {
    /// The array of pointer to the metadata container objects.
    struct metadataContainer** metadata_objects;

    /**
     * @brief An array to indicate the use state of each pointer in the @c metadata_objects array
     * A value of 1 indicates the pointer is in use and should have a reference count > 0
     */
    int* in_use;

    /// The size of the @c metadataContainer array.
    unsigned int pool_size;

    /// The size of the object stored by the metadata containers
    size_t metadata_object_size;

    /// Locks requests for metadata to avoid race conditions.
    pthread_mutex_t pool_lock;

    /// Name of the metadata pool
    char* unique_name;

    /// Data type of the metadata objects in this pool
    char* type_name;
};

/**
 * @brief Creates a new metadata pool with a fixed number of metadata containers.
 * @param[in] num_metadata_objects The number of containers to store in the pool.
 * @param[in] object_size The size of the actual metadata contained in each container.
 * @param[in] unique_name The name of the pool generated from the config path.
 * @param[in] type_name The data type name of the pool.
 * @return A metadata pool which can then be associated to one or more buffers.
 */
struct metadataPool* create_metadata_pool(int num_metadata_objects, size_t object_size,
                                          const char* unique_name, const char* type_name);

/**
 * @brief Deletes a memdata pool and frees all memory associated with its containers.
 * @param pool The pool to delete
 * @warning This should only be called once all metadata containers have been
 *          returned to the pool.  After the pipeline has shutdown.
 */
void delete_metadata_pool(struct metadataPool* pool);

/**
 * @brief Returns a metadata container with a reference count of 1.
 * @param[in] pool The pool to get the metadata object from.
 * @return A metadata container, or NULL if no containers are available.
 * @todo For now this asserts when unable to return a container, that should be fixed.
 */
struct metadataContainer* request_metadata_object(struct metadataPool* pool);

/**
 * @brief Returns a metadata container with a reference count of zero to its pool
 * @param[in] pool The pool to return the container too.
 * @param[in] container The container to return to the pool.
 */
void return_metadata_to_pool(struct metadataPool* pool, struct metadataContainer* container);

#ifdef __cplusplus
}
#endif

#endif
