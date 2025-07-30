#include <DataType.hpp>
#include <unordered_map>

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
    }
    return true;
}
static_assert(test_int4x2_t());

constexpr bool test_int4x2chime_t() {
    for (int val = 0x00; val <= 0xff; ++val) {
        const int v0 = ((val & 0x0f) >> 0) - 8;
        const int v1 = ((val & 0xf0) >> 4) - 8;
        if (!(-8 <= v0 && v0 <= +7))
            return false;
        if (!(-8 <= v1 && v1 <= +7))
            return false;
        const int4x2chime_t x(v0, v1);
        if (x.val != val)
            return false;
        const int4x2chime_t y{std::uint8_t(val)};
        if (y[0] != v0)
            return false;
        if (y[1] != v1)
            return false;
    }
    return true;
}
static_assert(test_int4x2chime_t());

} // namespace

static_assert(GetDataType_v<GetType_t<uint1x8>> == uint1x8);
static_assert(GetDataType_v<GetType_t<uint4x2>> == uint4x2);
static_assert(GetDataType_v<GetType_t<uint8>> == uint8);
static_assert(GetDataType_v<GetType_t<uint16>> == uint16);
static_assert(GetDataType_v<GetType_t<uint32>> == uint32);
static_assert(GetDataType_v<GetType_t<uint64>> == uint64);
static_assert(GetDataType_v<GetType_t<int4x2>> == int4x2);
static_assert(GetDataType_v<GetType_t<int4x2chime>> == int4x2chime);
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

const std::unordered_map<std::string, DataType> string_to_type_map{
    {"uint1x8", uint1x8},         //
    {"uint4x2", uint4x2},         //
    {"uint8", uint8},             //
    {"uint16", uint16},           //
    {"uint32", uint32},           //
    {"uint64", uint64},           //
    {"int4x2", int4x2},           //
    {"int4x2chime", int4x2chime}, //
    {"int8", int8},               //
    {"int16", int16},             //
    {"int32", int32},             //
    {"int64", int64},             //
    {"float16", float16},         //
    {"float32", float32},         //
    {"float64", float64},         //
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

} // namespace kotekan
