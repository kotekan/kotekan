#include "metadata.hpp"

#include "errors.h" // for CHECK_ERROR_F, CHECK_MEM_F

#include <assert.h> // for assert
#include <stdlib.h> // for malloc, free
#include <string.h> // for memset

metadataPool::metadataPool(Private, int num_metadata_objects, size_t object_size,
                           const std::string& _unique_name, const std::string& _type_name) :
    unique_name(_unique_name),
    type_name(_type_name),
    metadata_object_size(object_size),
    pool_size(num_metadata_objects)
{
}

metadataPool::~metadataPool() {
}

std::shared_ptr<metadataPool> metadataPool::create(int num_obj, size_t obj_size,
                                                   const std::string& unique_name, const std::string& type_name) {
    return std::make_shared<metadataPool>(Private(), num_obj, obj_size, unique_name, type_name);
}

std::shared_ptr<metadataObject> metadataPool::request_metadata_object() {
    auto t = FACTORY(metadataObject)::create_shared(type_name);
    t->parent_pool = get_weak();
    return t;
}

metadataObject::metadataObject() {
}

size_t metadataObject::get_object_size() {
    std::shared_ptr<metadataPool> pool = parent_pool.lock();
    if (!pool)
        FATAL_ERROR("metadataObject::get_object_size(): pool is null");
    return pool->metadata_object_size;
}
