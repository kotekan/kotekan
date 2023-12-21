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

#ifndef KOTEKAN_METADATA_HPP
#define KOTEKAN_METADATA_HPP

#include "factory.hpp"
#include "kotekanLogging.hpp"
#include "json.hpp" // for json

#include <memory>
#include <mutex>
#include <stdint.h> // for uint32_t
#include <stdio.h>  // for size_t
#include <vector>

class metadataPool;

// *** Metadata object section ***

class metadataObject : public kotekan::kotekanLogging {

public:
    metadataObject();
    virtual ~metadataObject() {}

    virtual void deepCopy(std::shared_ptr<metadataObject> other);

    /// Reference to metadataPool that this object belongs to.
    std::weak_ptr<metadataPool> parent_pool;

    /// Returns the size in memory of objects of this type, according to my metadataPool.
    size_t get_object_size();

    /// Returns the size of objects of this type when serialized into bytes.
    virtual size_t get_serialized_size() {
        return 0;
    }

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    virtual size_t set_from_bytes(const char* /*bytes*/, size_t /*length*/) {
        return 0;
    }

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    virtual size_t serialize(char* /*bytes*/) {
        return 0;
    }

    virtual nlohmann::json to_json();
};

void to_json(nlohmann::json& j, const metadataObject& m);
void from_json(const nlohmann::json& j, metadataObject& m);

CREATE_FACTORY(metadataObject);

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
class metadataPool : public kotekan::kotekanLogging,
                     public std::enable_shared_from_this<metadataPool> {
    // see "Best" example in https://en.cppreference.com/w/cpp/memory/enable_shared_from_this
    struct Private {};

public:
    // Constructor is only usable by this class
    metadataPool(Private, int num_metadata_objects, size_t object_size,
                 const std::string& unique_name, const std::string& type_name);
    ~metadataPool();
    // Everyone else has to use this factory function
    // Hence all metadataPool objects are managed by shared_ptrs
    static std::shared_ptr<metadataPool> create(int num_metadata_objects, size_t object_size,
                                                const std::string& unique_name,
                                                const std::string& type_name);

    std::shared_ptr<metadataPool> get_shared() {
        return shared_from_this();
    }
    std::weak_ptr<metadataPool> get_weak() {
        return weak_from_this();
    }

    std::shared_ptr<metadataObject> request_metadata_object();

    // void return_metadata_to_pool(struct metadataPool* pool, std::shared_ptr<metadataObject>
    // container);

    /// Name of the metadata pool
    std::string unique_name;

    /// Data type of the metadata objects in this pool
    std::string type_name;

    /// The size of the object stored in this pool
    size_t metadata_object_size;

protected:
    /// The underlying block af data that we allocate objects out of
    void* data_block;

    /**
     * @brief An array to indicate the use state of each pointer in the @c metadata_objects array
     * A value of 1 indicates the pointer is in use and should have a reference count > 0
     */
    std::vector<bool> in_use;

    /// The size of the @c metadataContainer array.
    unsigned int pool_size;

    /// Locks requests for metadata to avoid race conditions.
    std::mutex pool_mutex;
};

#endif
