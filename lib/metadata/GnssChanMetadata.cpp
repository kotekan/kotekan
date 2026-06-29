#include "GnssChanMetadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <cstring> // for memcpy

REGISTER_TYPE_WITH_FACTORY(metadataObject, GnssChanMetadata);

void GnssChanMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto o = std::dynamic_pointer_cast<const GnssChanMetadata>(other);
    if (o)
        sample_seq = o->sample_seq;
}

size_t GnssChanMetadata::get_serialized_size() {
    return sizeof(int64_t);
}

size_t GnssChanMetadata::serialize(char* bytes) {
    std::memcpy(bytes, &sample_seq, sizeof(int64_t));
    return sizeof(int64_t);
}

size_t GnssChanMetadata::set_from_bytes(const char* bytes, size_t length) {
    (void)length;
    std::memcpy(&sample_seq, bytes, sizeof(int64_t));
    return sizeof(int64_t);
}

nlohmann::json GnssChanMetadata::to_json() {
    return {{"sample_seq", sample_seq}};
}
