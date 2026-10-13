#include "GnssChanMetadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <algorithm> // for min
#include <cstring>   // for memcpy

REGISTER_TYPE_WITH_FACTORY(metadataObject, GnssChanMetadata);

void GnssChanMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto o = std::dynamic_pointer_cast<const GnssChanMetadata>(other);
    if (o)
        sample_seq = o->sample_seq;
}

// Wire format: [int64 sample_seq][uint32 0]. The zero word is reserved; it keeps the 12-byte
// layout that bufferSend/bufferRecv peers and the cube archive's raw files expect.
size_t GnssChanMetadata::get_serialized_size() {
    return sizeof(int64_t) + sizeof(uint32_t);
}

size_t GnssChanMetadata::serialize(char* bytes) {
    std::memcpy(bytes, &sample_seq, sizeof(int64_t));
    const uint32_t reserved = 0;
    std::memcpy(bytes + sizeof(int64_t), &reserved, sizeof(uint32_t));
    return get_serialized_size();
}

size_t GnssChanMetadata::set_from_bytes(const char* bytes, size_t length) {
    std::memcpy(&sample_seq, bytes, sizeof(int64_t));
    return std::min(get_serialized_size(), length);
}

nlohmann::json GnssChanMetadata::to_json() {
    return {{"sample_seq", sample_seq}};
}
