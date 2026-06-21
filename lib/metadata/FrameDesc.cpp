#include "FrameDesc.hpp"

#include <cstdint>     // for uint32_t
#include <cstring>     // for memcpy
#include <stdexcept>   // for runtime_error

#include "N2FrameDesc.hpp"  // for N2FrameDesc
#include "NDArray.hpp"      // for GenericNDArray
#include "fmt.hpp"          // for format

namespace kotekan {

std::shared_ptr<const FrameDesc> FrameDesc::deserialize(const char* bytes, size_t size) {
    if (size < sizeof(uint32_t))
        throw std::runtime_error("FrameDesc::deserialize: input too small for the wire tag");

    uint32_t tag;
    std::memcpy(&tag, bytes, sizeof(tag));
    const char* payload = bytes + sizeof(tag);
    const size_t payload_size = size - sizeof(tag);

    switch (static_cast<WireType>(tag)) {
        case WireType::generic_ndarray:
            return GenericNDArray::deserialize_payload(payload, payload_size);
        case WireType::n2:
            return N2FrameDesc::deserialize_payload(payload, payload_size);
        default:
            throw std::runtime_error(
                fmt::format(fmt("FrameDesc::deserialize: unknown wire_type tag {:d}"), tag));
    }
}

} // namespace kotekan
