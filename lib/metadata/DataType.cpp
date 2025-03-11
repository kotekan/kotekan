#include <DataType.hpp>

namespace kotekan {

std::string type_to_string(DataType type) {
    switch (type) {
        case unknown_type:
            return "unknown_type";
        case uint1x8:
            return "uint1x8";
        case uint4x2:
            return "uint4x2";
        case uint8:
            return "uint8";
        case uint16:
            return "uint16";
        case uint32:
            return "uint32";
        case uint64:
            return "uint64";
        case int4x2:
            return "int4x2";
        case int4x2chime:
            return "int4x2chime";
        case int8:
            return "int8";
        case int16:
            return "int16";
        case int32:
            return "int32";
        case int64:
            return "int64";
        case float16:
            return "float16";
        case float32:
            return "float32";
        case float64:
            return "float64";
        default:
            return "error_type";
    }
}

std::ostream& operator<<(std::ostream& os, DataType type) {
    return os << type_to_string(type);
}

} // namespace kotekan
