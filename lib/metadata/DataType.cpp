#include <DataType.hpp>
#include <unordered_map> // for unordered_map, operator==, _Node_iterator_base

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
        case int4x2_swapped_withoffset:
            return "int4x2_swapped_withoffset";
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
        case cuint8:
            return "cuint8";
        case cuint16:
            return "cuint16";
        case cuint32:
            return "cuint32";
        case cuint64:
            return "cuint64";
        case cint8:
            return "cint8";
        case cint16:
            return "cint16";
        case cint32:
            return "cint32";
        case cint64:
            return "cint64";
        case cfloat16:
            return "cfloat16";
        case cfloat32:
            return "cfloat32";
        case cfloat64:
            return "cfloat64";
        default:
            return "error_type";
    }
}

const std::unordered_map<std::string, DataType> string_to_type_map{
    {"uint1x8", uint1x8},                                     //
    {"uint4x2", uint4x2},                                     //
    {"uint8", uint8},                                         //
    {"uint16", uint16},                                       //
    {"uint32", uint32},                                       //
    {"uint64", uint64},                                       //
    {"int4x2", int4x2},                                       //
    {"int4x2_swapped_withoffset", int4x2_swapped_withoffset}, //
    {"int8", int8},                                           //
    {"int16", int16},                                         //
    {"int32", int32},                                         //
    {"int64", int64},                                         //
    {"float16", float16},                                     //
    {"float32", float32},                                     //
    {"float64", float64},                                     //
    {"cuint8", cuint8},                                       //
    {"cuint16", cuint16},                                     //
    {"cuint32", cuint32},                                     //
    {"cuint64", cuint64},                                     //
    {"cint8", cint8},                                         //
    {"cint16", cint16},                                       //
    {"cint32", cint32},                                       //
    {"cint64", cint64},                                       //
    {"cfloat16", cfloat16},                                   //
    {"cfloat32", cfloat32},                                   //
    {"cfloat64", cfloat64},                                   //
};

DataType string_to_type(const std::string& type_name) {
    const auto iter = string_to_type_map.find(type_name);
    if (iter == string_to_type_map.end())
        return unknown_type;
    return iter->second;
}

std::ostream& operator<<(std::ostream& os, DataType type) {
    return os << type_to_string(type);
}

void to_json(nlohmann::json& j, const DataType& d) {
    j = type_to_string(d);
}

void from_json(const nlohmann::json& j, DataType& d) {
    d = string_to_type(j.get<std::string>());
}

} // namespace kotekan
