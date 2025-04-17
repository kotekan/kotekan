#include "gdalFiles.hpp"

namespace gdal {

GDALDataType chord2gdal(const kotekan::DataType type) {
    switch (type) {
        case kotekan::uint1x8: // TODO: Define GDAL bool8 type
            return GDT_Byte;
        case kotekan::uint4x2:
            return GDT_Byte; // TODO: Define GDAL uint4+4 type
        case kotekan::uint8:
            return GDT_Byte;
        case kotekan::uint16:
            return GDT_UInt16;
        case kotekan::uint32:
            return GDT_UInt32;
        case kotekan::uint64:
            return GDT_UInt64;
        case kotekan::int4x2:
            return GDT_Byte; // TODO: Define GDAL int4+4 type
        case kotekan::int4x2chime:
            return GDT_Byte; // TODO: Define GDAL int4+4 type
        case kotekan::int8:
            return GDT_Int8;
        case kotekan::int16:
            return GDT_Int16;
        case kotekan::int32:
            return GDT_Int32;
        case kotekan::int64:
            return GDT_Int64;
        case kotekan::float16:
            return GDT_UInt16; // TODO: Define GDAL float16 type
        case kotekan::float32:
            return GDT_Float32;
        case kotekan::float64:
            return GDT_Float64;
        default:
            return GDT_Unknown;
    }
}

std::vector<const char*> convert_to_cstring_list(const std::vector<std::string>& strings) {
    std::vector<const char*> result;
    // Convert strings to C strings
    for (const auto& str : strings)
        result.push_back(str.c_str());
    // Add trailing NULL
    result.push_back(nullptr);
    return result;
}

} // namespace gdal
