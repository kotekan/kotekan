#include "hdf5Files.hpp"

#include <cassert>
#include <cstdint>

namespace hdf5 {

HighFive::DataType chord2hdf5(const kotekan::DataType type) {
    switch (type) {
        case kotekan::uint1x8:
            return HighFive::AtomicType<std::uint8_t>(); // 8 bools (packed into a type)
        case kotekan::uint4x2:
            return HighFive::AtomicType<std::uint8_t>(); // 2 unsigned 4-bit integers (packed into a
                                                         // byte)
        case kotekan::uint8:
            return HighFive::AtomicType<std::uint8_t>();
        case kotekan::uint16:
            return HighFive::AtomicType<std::uint16_t>();
        case kotekan::uint32:
            return HighFive::AtomicType<std::uint32_t>();
        case kotekan::uint64:
            return HighFive::AtomicType<std::uint64_t>();
        case kotekan::int4x2:
            return HighFive::AtomicType<std::uint8_t>(); // 2 signed 4-bit integers (packed into a
                                                         // byte)
        case kotekan::int4x2_swapped_withoffset:
            return HighFive::AtomicType<std::uint8_t>(); // offset-encoded (stored is value + 8),
                                                         // low and high values swapped
        case kotekan::int8:
            return HighFive::AtomicType<std::int8_t>();
        case kotekan::int16:
            return HighFive::AtomicType<std::int16_t>();
        case kotekan::int32:
            return HighFive::AtomicType<std::int32_t>();
        case kotekan::int64:
            return HighFive::AtomicType<std::int64_t>();
        case kotekan::float16:
            return HighFive::AtomicType<float16_t>();
            // return HighFive::AtomicType<std::uint16_t>();
        case kotekan::float32:
            return HighFive::AtomicType<float>();
        case kotekan::float64:
            return HighFive::AtomicType<double>();
        default:
            assert(0);
    }
}

} // namespace hdf5
