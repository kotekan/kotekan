#include "GnssChanMetadata.hpp"

#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <cstring> // for memcpy

REGISTER_TYPE_WITH_FACTORY(metadataObject, GnssChanMetadata);

void GnssChanMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto o = std::dynamic_pointer_cast<const GnssChanMetadata>(other);
    if (o)
        fpga_seq = o->fpga_seq;
}

size_t GnssChanMetadata::get_serialized_size() {
    return sizeof(int64_t);
}

size_t GnssChanMetadata::serialize(char* bytes) {
    std::memcpy(bytes, &fpga_seq, sizeof(int64_t));
    return sizeof(int64_t);
}

size_t GnssChanMetadata::set_from_bytes(const char* bytes, size_t length) {
    (void)length;
    std::memcpy(&fpga_seq, bytes, sizeof(int64_t));
    return sizeof(int64_t);
}

nlohmann::json GnssChanMetadata::to_json() {
    return {{"fpga_seq", fpga_seq}};
}
