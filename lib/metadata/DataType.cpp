#include <DataType.hpp>

std::ostream& operator<<(std::ostream& os, DataType type) {
    switch (type) {
        case unknown_type:
            return os << "unknown_type";
        case uint4p4:
            return os << "uint4p4";
        case uint8:
            return os << "uint8";
        case uint16:
            return os << "uint16";
        case uint32:
            return os << "uint32";
        case uint64:
            return os << "uint64";
        case int4p4:
            return os << "int4p4";
        case int4p4chime:
            return os << "int4p4chime";
        case int8:
            return os << "int8";
        case int16:
            return os << "int16";
        case int32:
            return os << "int32";
        case int64:
            return os << "int64";
        case float16:
            return os << "float16";
        case float32:
            return os << "float32";
        case float64:
            return os << "float64";
        default:
            return os << "error_type";
    }
}
