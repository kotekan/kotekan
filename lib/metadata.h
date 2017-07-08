#ifndef BUFFER_INFO
#define BUFFER_INFO

#ifdef __cplusplus
extern "C" {
#endif

// *** Metadata object section ***

/** @brief Object containing metadata about a given buffer.
 *
 */
struct metadataContainer {

    void * metadata;
    size_t metadata_size;

    uint32_t ref_count;

    // Lock for variables in this scope.
    pthread_mutex_t metadata_lock;

    // Avoids passing the pool to some of the functions.
    struct metadataPool * parent_pool;
};

struct metadataContainer * create_metadata(size_t object_size, struct metadataPool * parent_pool);

void delete_metadata(struct metadataContainer * container);

void reset_metadata_object(struct metadataContainer * container);

// Do NOT call if there is a lock on the metadata open in the calling function.
void increment_metadata_ref_count(struct metadataContainer * container);
void decrement_metadata_ref_count(struct metadataContainer * container);

// Needed for meta data that can be changed by more than one process.
inline void lock_metadata(struct metadataContainer * container);
inline void unlock_metadata(struct metadataContainer * container);

// *** Metadata pool section ***

struct metadataPool {
    struct metadataContainer ** metadata_objects;
    int * in_use;

    unsigned int pool_size;

    // Prevents us from allocating the same info object twice
    pthread_mutex_t pool_lock;
};

struct metadataPool * create_metadata_pool(struct metadataPool * pool, int num_metadata_objects, size_t object_size);

void delete_metadata_pool(struct metadataPool * pool);

// Returns a info object with a ref count of 1.
struct metadataContainer * request_metadata_object(struct metadataPool * pool);

void return_metadata_to_pool(struct metadataPool * pool, struct metadataContainer * info);

#ifdef __cplusplus
}
#endif

#endif