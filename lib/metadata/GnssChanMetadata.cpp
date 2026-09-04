#include "GnssChanMetadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <algorithm> // for min
#include <cstring>   // for memcpy

REGISTER_TYPE_WITH_FACTORY(metadataObject, GnssChanMetadata);

void GnssChanMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto o = std::dynamic_pointer_cast<const GnssChanMetadata>(other);
    if (o) {
        sample_seq = o->sample_seq;
        chan_scale = o->chan_scale;
    }
}

// Wire format: [int64 sample_seq][uint32 n_scale][float x n_scale]. An 8-byte blob (the format
// before chan_scale existed) still parses -- n_scale reads as absent, chan_scale comes back
// empty, which already means "fp32 / no scales".
size_t GnssChanMetadata::get_serialized_size() {
    return sizeof(int64_t) + sizeof(uint32_t) + chan_scale.size() * sizeof(float);
}

size_t GnssChanMetadata::serialize(char* bytes) {
    std::memcpy(bytes, &sample_seq, sizeof(int64_t));
    const uint32_t n = (uint32_t)chan_scale.size();
    std::memcpy(bytes + sizeof(int64_t), &n, sizeof(uint32_t));
    if (n)
        std::memcpy(bytes + sizeof(int64_t) + sizeof(uint32_t), chan_scale.data(),
                    n * sizeof(float));
    return get_serialized_size();
}

size_t GnssChanMetadata::set_from_bytes(const char* bytes, size_t length) {
    std::memcpy(&sample_seq, bytes, sizeof(int64_t));
    chan_scale.clear();
    if (length < sizeof(int64_t) + sizeof(uint32_t))
        return sizeof(int64_t); // pre-chan_scale blob
    uint32_t n = 0;
    std::memcpy(&n, bytes + sizeof(int64_t), sizeof(uint32_t));
    const size_t need = sizeof(int64_t) + sizeof(uint32_t) + (size_t)n * sizeof(float);
    if (n && length >= need) {
        chan_scale.resize(n);
        std::memcpy(chan_scale.data(), bytes + sizeof(int64_t) + sizeof(uint32_t),
                    (size_t)n * sizeof(float));
    }
    return std::min(need, length);
}

nlohmann::json GnssChanMetadata::to_json() {
    return {{"sample_seq", sample_seq}, {"n_chan_scale", chan_scale.size()}};
}
