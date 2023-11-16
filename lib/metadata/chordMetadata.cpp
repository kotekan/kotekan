#include "chordMetadata.hpp"

const char* chord_datatype_string(chordDataType type) {
    switch (type) {
        case int4p4:
            return "int4p4";
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
        case unknown_type:
        default:
            return "<unknown-type>";
    }
}
