#include <DataType.hpp>
#include <unordered_map> // for unordered_map, operator==, _Node_iterator_base
#include <utility>       // for pair

namespace kotekan {

namespace {

// Test the low-bit data types exhaustively at compile time

constexpr bool test_uint1x8_t() {
    for (int val = 0x00; val <= 0xff; ++val) {
        const bool v0 = val & 0x01;
        const bool v1 = val & 0x02;
        const bool v2 = val & 0x04;
        const bool v3 = val & 0x08;
        const bool v4 = val & 0x10;
        const bool v5 = val & 0x20;
        const bool v6 = val & 0x40;
        const bool v7 = val & 0x80;
        const uint1x8_t x(v0, v1, v2, v3, v4, v5, v6, v7);
        if (x.val != val)
            return false;
        const uint1x8_t y{std::uint8_t(val)};
        if (y[0] != v0)
            return false;
        if (y[1] != v1)
            return false;
        if (y[2] != v2)
            return false;
        if (y[3] != v3)
            return false;
        if (y[4] != v4)
            return false;
        if (y[5] != v5)
            return false;
        if (y[6] != v6)
            return false;
        if (y[7] != v7)
            return false;
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const bool w0 = val2 & 0x01;
            const bool w1 = val2 & 0x02;
            const bool w2 = val2 & 0x04;
            const bool w3 = val2 & 0x08;
            const bool w4 = val2 & 0x10;
            const bool w5 = val2 & 0x20;
            const bool w6 = val2 & 0x40;
            const bool w7 = val2 & 0x80;
            const uint1x8_t x2(w0, w1, w2, w3, w4, w5, w6, w7);
            if ((x2 == x) != (val == val2))
                return false;
            if ((x2 != x) != (val != val2))
                return false;
        }
    }
    return true;
}
static_assert(test_uint1x8_t());

constexpr bool test_uint4x2_t() {
    for (int val = 0x00; val <= 0xff; ++val) {
        const unsigned int v0 = (val & 0x0f) >> 0;
        const unsigned int v1 = (val & 0xf0) >> 4;
        if (!(v0 <= 15))
            return false;
        if (!(v1 <= 15))
            return false;
        const uint4x2_t x(v0, v1);
        if (x.val != val)
            return false;
        const uint4x2_t y{std::uint8_t(val)};
        if (y[0] != v0)
            return false;
        if (y[1] != v1)
            return false;
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const unsigned int w0 = (val2 & 0x0f) >> 0;
            const unsigned int w1 = (val2 & 0xf0) >> 4;
            const uint4x2_t x2(w0, w1);
            if ((x2 == x) != (val == val2))
                return false;
            if ((x2 != x) != (val != val2))
                return false;
        }
    }
    return true;
}
static_assert(test_uint4x2_t());

constexpr bool test_int4x2_t() {
    for (int val = 0x00; val <= 0xff; ++val) {
        const int v0 = std::int8_t(((val & 0x0f) >> 0) << 4) >> 4;
        const int v1 = std::int8_t(((val & 0xf0) >> 4) << 4) >> 4;
        if (!(-8 <= v0 && v0 <= +7))
            return false;
        if (!(-8 <= v1 && v1 <= +7))
            return false;
        const int4x2_t x(v0, v1);
        if (x.val != val)
            return false;
        const int4x2_t y{std::uint8_t(val)};
        if (y[0] != v0)
            return false;
        if (y[1] != v1)
            return false;
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const int w0 = std::int8_t(((val2 & 0x0f) >> 0) << 4) >> 4;
            const int w1 = std::int8_t(((val2 & 0xf0) >> 4) << 4) >> 4;
            const int4x2_t x2(w0, w1);
            if ((x2 == x) != (val == val2))
                return false;
            if ((x2 != x) != (val != val2))
                return false;
        }
    }
    return true;
}
static_assert(test_int4x2_t());

constexpr bool test_int4x2_swapped_withoffset_t() {
    for (int val = 0x00; val <= 0xff; ++val) {
        const int v0 = ((val & 0x0f) >> 0) - 8;
        const int v1 = ((val & 0xf0) >> 4) - 8;
        if (!(-8 <= v0 && v0 <= +7))
            return false;
        if (!(-8 <= v1 && v1 <= +7))
            return false;
        const int4x2_swapped_withoffset_t x(v0, v1);
        if (x.val != val)
            return false;
        const int4x2_swapped_withoffset_t y{std::uint8_t(val)};
        if (y[0] != v0)
            return false;
        if (y[1] != v1)
            return false;
        for (int val2 = 0x00; val2 <= 0xff; ++val2) {
            const int w0 = ((val2 & 0x0f) >> 0) - 8;
            const int w1 = ((val2 & 0xf0) >> 4) - 8;
            const int4x2_swapped_withoffset_t x2(w0, w1);
            if ((x2 == x) != (val == val2))
                return false;
            if ((x2 != x) != (val != val2))
                return false;
        }
    }
    return true;
}
static_assert(test_int4x2_swapped_withoffset_t());

} // namespace

static_assert(GetDataType_v<GetType_t<uint1x8>> == uint1x8);
static_assert(GetDataType_v<GetType_t<uint4x2>> == uint4x2);
static_assert(GetDataType_v<GetType_t<uint8>> == uint8);
static_assert(GetDataType_v<GetType_t<uint16>> == uint16);
static_assert(GetDataType_v<GetType_t<uint32>> == uint32);
static_assert(GetDataType_v<GetType_t<uint64>> == uint64);
static_assert(GetDataType_v<GetType_t<int4x2>> == int4x2);
static_assert(GetDataType_v<GetType_t<int4x2_swapped_withoffset>> == int4x2_swapped_withoffset);
static_assert(GetDataType_v<GetType_t<int8>> == int8);
static_assert(GetDataType_v<GetType_t<int16>> == int16);
static_assert(GetDataType_v<GetType_t<int32>> == int32);
static_assert(GetDataType_v<GetType_t<int64>> == int64);
static_assert(GetDataType_v<GetType_t<float16>> == float16);
static_assert(GetDataType_v<GetType_t<float32>> == float32);
static_assert(GetDataType_v<GetType_t<float64>> == float64);

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
