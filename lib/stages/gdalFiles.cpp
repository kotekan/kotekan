#include "gdalFiles.hpp"

namespace gdal {

GDALDataType chord2gdal(const chordDataType type) {
    switch (type) {
        case uint4p4:
            return GDT_Byte; // TODO: Define GDAL uint4+4 type
        case uint8:
            return GDT_Byte;
        case uint16:
            return GDT_UInt16;
        case uint32:
            return GDT_UInt32;
        case uint64:
            return GDT_UInt64;
        case int4p4:
            return GDT_Byte; // TODO: Define GDAL int4+4 type
        case int4p4chime:
            return GDT_Byte; // TODO: Define GDAL int4+4 type
        case int8:
            return GDT_Int8;
        case int16:
            return GDT_Int16;
        case int32:
            return GDT_Int32;
        case int64:
            return GDT_Int64;
        case float16:
            return GDT_UInt16; // TODO: Define GDAL float16 type
        case float32:
            return GDT_Float32;
        case float64:
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
