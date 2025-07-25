#ifndef ASDFFILES_HPP
#define ASDFFILES_HPP

#include <DataType.hpp>
#include <array>
#include <asdf/asdf.hxx>
#include <regex>
#include <string>

namespace asdf {

// The version number of the CHORD metadata format in ASDF files.
// This uses semver: A file with version X.Y can be read by software
// that understands X.Z even when Y > Z.
constexpr std::array<int, 2> chord_metadata_version{1, 0};

inline ASDF::scalar_type_id_t chord2asdf(const kotekan::DataType type) {
    switch (type) {
        case kotekan::uint1x8:
            return ASDF::id_uint8; // TODO: Define ASDF bool8 type
        case kotekan::uint4x2:
            return ASDF::id_uint8; // TODO: Define ASDF uint4+4 type
        case kotekan::uint8:
            return ASDF::id_uint8;
        case kotekan::uint16:
            return ASDF::id_uint16;
        case kotekan::uint32:
            return ASDF::id_uint32;
        case kotekan::uint64:
            return ASDF::id_uint64;
        case kotekan::int4x2:
            return ASDF::id_uint8; // TODO: Define ASDF int4+4 type
        case kotekan::int4x2chime:
            return ASDF::id_uint8; // TODO: Define ASDF int4+4 type
        case kotekan::int8:
            return ASDF::id_int8;
        case kotekan::int16:
            return ASDF::id_int16;
        case kotekan::int32:
            return ASDF::id_int32;
        case kotekan::int64:
            return ASDF::id_int64;
        case kotekan::float16:
            return ASDF::id_float16;
        case kotekan::float32:
            return ASDF::id_float32;
        case kotekan::float64:
            return ASDF::id_float64;
        default:
            assert(0);
    }
}

inline std::string beautify_buffer_name(const std::string& name) {
    // Remove "_host_" prefix and/or "_buffer" suffix
    const std::regex pattern(R"(^(host_)?(.*?)(_buffer)?$)");
    std::smatch match;
    if (std::regex_match(name, match, pattern))
        return match[2];
    return name;
}

} // namespace asdf

#endif // ASDFFILES_HPP
