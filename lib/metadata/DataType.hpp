#ifndef DATATYPE_HPP
#define DATATYPE_HPP

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>
#include <type_traits>

// Define a macro `KOTEKAN_FLOAT16` specifying whether we support a float16 type.
// Define a type `float16_t` if we support it.

#if defined WITH_CUDA
// If we use CUDA, use its float16 type
#include <cuda_fp16.h>
using float16_t = __half;
#define KOTEKAN_FLOAT16 1
#else
// If we don't use CUDA, see whether the compiler supports it
#include <float.h>
#if defined __FLT16_MAX__
using float16_t = _Float16;
#define KOTEKAN_FLOAT16 1
#else
// There is no float16 type
#define KOTEKAN_FLOAT16 0
#endif
#endif

namespace kotekan {

#ifdef DEBUGGING
#define KOTEKAN_ASSERT(cond) assert(cond)
#else
#define KOTEKAN_ASSERT(cond) 0
#endif

// 8 bools (packed into a type)
struct uint1x8_t {
    std::uint8_t val;

    uint1x8_t() = default;
    uint1x8_t(const uint1x8_t&) = default;
    uint1x8_t(uint1x8_t&&) = default;
    uint1x8_t& operator=(const uint1x8_t&) = default;
    uint1x8_t& operator=(uint1x8_t&&) = default;
    constexpr uint1x8_t(std::initializer_list<std::uint8_t> vals) :
        val((KOTEKAN_ASSERT(vals.size() == 1), *vals.begin())) {}

    constexpr uint1x8_t(bool v0, bool v1, bool v2, bool v3, bool v4, bool v5, bool v6, bool v7) :
        val(((v0 & 1) << 0) | ((v1 & 1) << 1) | ((v2 & 1) << 2) | ((v3 & 1) << 3) | ((v4 & 1) << 4)
            | ((v5 & 1) << 5) | ((v6 & 1) << 6) | ((v7 & 1) << 7)) {}
    constexpr bool operator[](int n) const {
        KOTEKAN_ASSERT(0 <= n && n < 8);
        return (val >> n) & 1;
    }
};

// 2 unsigned 4-bit integers (packed into a byte)
struct uint4x2_t {
    std::uint8_t val;

    uint4x2_t() = default;
    uint4x2_t(const uint4x2_t&) = default;
    uint4x2_t(uint4x2_t&&) = default;
    uint4x2_t& operator=(const uint4x2_t&) = default;
    uint4x2_t& operator=(uint4x2_t&&) = default;
    constexpr uint4x2_t(std::initializer_list<std::uint8_t> vals) :
        val((KOTEKAN_ASSERT(vals.size() == 1), *vals.begin())) {}

    constexpr uint4x2_t(std::uint8_t v0, std::uint8_t v1) :
        val(((v0 & 0xf) << 0) | ((v1 & 0xf) << 4)) {}
    constexpr std::uint8_t operator[](int n) const {
        KOTEKAN_ASSERT(0 <= n && n < 2);
        return (val >> (4 * n)) & 0xf;
    }
};

// 2 signed 4-bit integers (packed into a byte)
struct int4x2_t {
    std::uint8_t val;

    int4x2_t() = default;
    int4x2_t(const int4x2_t&) = default;
    int4x2_t(int4x2_t&&) = default;
    int4x2_t& operator=(const int4x2_t&) = default;
    int4x2_t& operator=(int4x2_t&&) = default;
    constexpr int4x2_t(std::initializer_list<std::uint8_t> vals) :
        val((KOTEKAN_ASSERT(vals.size() == 1), *vals.begin())) {}

    constexpr int4x2_t(std::int8_t v0, std::int8_t v1) :
        val(((v0 & 0xf) << 0) | ((v1 & 0xf) << 4)) {}
    constexpr std::int8_t operator[](int n) const {
        KOTEKAN_ASSERT(0 <= n && n < 2);
        const int bits = (val >> (4 * n)) & 0xf;
        return (std::int8_t)(bits << 4) >> 4;
    }
};

// offset-encoded (stored is value + 8), low and high values swapped
struct int4x2chime_t {
    std::uint8_t val;

    int4x2chime_t() = default;
    int4x2chime_t(const int4x2chime_t&) = default;
    int4x2chime_t(int4x2chime_t&&) = default;
    int4x2chime_t& operator=(const int4x2chime_t&) = default;
    int4x2chime_t& operator=(int4x2chime_t&&) = default;
    constexpr int4x2chime_t(std::initializer_list<std::uint8_t> vals) :
        val((KOTEKAN_ASSERT(vals.size() == 1), *vals.begin())) {}

    constexpr int4x2chime_t(std::int8_t v0, std::int8_t v1) :
        val((((unsigned)(v0 + 8) & 0xf) << 0) | (((unsigned)(v1 + 8) & 0xf) << 4)) {}
    constexpr std::int8_t operator[](int n) const {
        KOTEKAN_ASSERT(0 <= n && n < 2);
        const int bits = (val >> (4 * n)) & 0xf;
        return bits - 8;
    }
};

#undef KOTEKAN_ASSERT

// This enum lets us talk about the various datatypes we're using.
enum DataType {
    unknown_type,
    uint1x8, // 8 bools (packed into a type)
    uint4x2, // 2 unsigned 4-bit integers (packed into a byte)
    uint8,
    uint16,
    uint32,
    uint64,
    int4x2,      // 2 signed 4-bit integers (packed into a byte)
    int4x2chime, // offset-encoded (stored is value + 8), low and high values swapped
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
};

// Number of bits (not bytes!) in a type. For packed types, say how
// many bits there are in each element.
constexpr std::size_t type_value_bits(DataType type) {
    switch (type) {
        case uint1x8:
            return 1;
        case uint4x2:
            return 4;
        case uint8:
            return 8;
        case uint16:
            return 16;
        case uint32:
            return 32;
        case uint64:
            return 64;
        case int4x2:
            return 4;
        case int4x2chime:
            return 4;
        case int8:
            return 8;
        case int16:
            return 16;
        case int32:
            return 32;
        case int64:
            return 64;
        case float16:
            return 16;
        case float32:
            return 32;
        case float64:
            return 64;
        default:
            return -1;
    }
}

// Number of bytes (not bits!) in a type. For packed types this is the
// total size for all elements. element.
constexpr std::size_t type_total_bytes(DataType type) {
    switch (type) {
        case uint1x8:
            return 1;
        case uint4x2:
            return 1;
        case uint8:
            return 1;
        case uint16:
            return 2;
        case uint32:
            return 4;
        case uint64:
            return 8;
        case int4x2:
            return 1;
        case int4x2chime:
            return 1;
        case int8:
            return 1;
        case int16:
            return 2;
        case int32:
            return 4;
        case int64:
            return 8;
        case float16:
            return 2;
        case float32:
            return 4;
        case float64:
            return 8;
        default:
            return -1;
    }
}

// Find an unsigned int with that many bits
constexpr DataType uint_from_element_bits(std::size_t bits) {
    switch (bits) {
        case 1:
            return uint1x8;
        case 4:
            return uint4x2;
        case 8:
            return uint8;
        case 16:
            return uint16;
        case 32:
            return uint32;
        case 64:
            return uint64;
        default:
            return unknown_type;
    }
}

// Find a signed int with that many bits
constexpr DataType int_from_element_bits(std::size_t bits) {
    switch (bits) {
        case 4:
            return int4x2;
        case 8:
            return int8;
        case 16:
            return int16;
        case 32:
            return int32;
        case 64:
            return int64;
        default:
            return unknown_type;
    }
}

// Find a floating-point type with that many bits
constexpr DataType float_from_element_bits(std::size_t bits) {
    switch (bits) {
        case 16:
            return float16;
        case 32:
            return float32;
        case 64:
            return float64;
        default:
            return unknown_type;
    }
}

// Convert a C++ type to a type enum
template<typename T>
struct GetDataType;

template<>
struct GetDataType<unsigned char>
    : std::integral_constant<DataType, uint_from_element_bits(8 * sizeof(unsigned char))> {};
template<>
struct GetDataType<unsigned short>
    : std::integral_constant<DataType, uint_from_element_bits(8 * sizeof(unsigned short))> {};
template<>
struct GetDataType<unsigned int>
    : std::integral_constant<DataType, uint_from_element_bits(8 * sizeof(unsigned int))> {};
template<>
struct GetDataType<unsigned long>
    : std::integral_constant<DataType, uint_from_element_bits(8 * sizeof(unsigned long))> {};
template<>
struct GetDataType<unsigned long long>
    : std::integral_constant<DataType, uint_from_element_bits(8 * sizeof(unsigned long long))> {};
template<>
struct GetDataType<signed char>
    : std::integral_constant<DataType, int_from_element_bits(8 * sizeof(signed char))> {};
template<>
struct GetDataType<char>
    : std::integral_constant<DataType, (std::is_signed_v<char>
                                            ? int_from_element_bits(8 * sizeof(char))
                                            : uint_from_element_bits(8 * sizeof(char)))> {};
template<>
struct GetDataType<short>
    : std::integral_constant<DataType, int_from_element_bits(8 * sizeof(short))> {};
template<>
struct GetDataType<int> : std::integral_constant<DataType, int_from_element_bits(8 * sizeof(int))> {
};
template<>
struct GetDataType<long>
    : std::integral_constant<DataType, int_from_element_bits(8 * sizeof(long))> {};
template<>
struct GetDataType<long long>
    : std::integral_constant<DataType, int_from_element_bits(8 * sizeof(long long))> {};
#if KOTEKAN_FLOAT16
template<>
struct GetDataType<float16_t>
    : std::integral_constant<DataType, float_from_element_bits(8 * sizeof(float16_t))> {};
#endif
template<>
struct GetDataType<float>
    : std::integral_constant<DataType, float_from_element_bits(8 * sizeof(float))> {};
template<>
struct GetDataType<double>
    : std::integral_constant<DataType, float_from_element_bits(8 * sizeof(double))> {};
template<>
struct GetDataType<long double>
    : std::integral_constant<DataType, float_from_element_bits(8 * sizeof(long double))> {};

template<>
struct GetDataType<uint1x8_t> : std::integral_constant<DataType, uint1x8> {};
template<>
struct GetDataType<uint4x2_t> : std::integral_constant<DataType, uint4x2> {};
template<>
struct GetDataType<int4x2_t> : std::integral_constant<DataType, int4x2> {};
template<>
struct GetDataType<int4x2chime_t> : std::integral_constant<DataType, int4x2chime> {};

// Use e.g. as `DataType double_type = GetDataType_v<double>`
template<typename T>
constexpr DataType GetDataType_v = GetDataType<T>::value;

template<DataType type>
struct GetType;
template<>
struct GetType<uint1x8> {
    using type = uint1x8_t;
};
template<>
struct GetType<uint4x2> {
    using type = uint4x2_t;
};
template<>
struct GetType<uint8> {
    using type = std::uint8_t;
};
template<>
struct GetType<uint16> {
    using type = std::uint16_t;
};
template<>
struct GetType<uint32> {
    using type = std::uint32_t;
};
template<>
struct GetType<uint64> {
    using type = std::uint64_t;
};
template<>
struct GetType<int4x2> {
    using type = int4x2_t;
};
template<>
struct GetType<int4x2chime> {
    using type = int4x2chime_t;
};
template<>
struct GetType<int8> {
    using type = std::int8_t;
};
template<>
struct GetType<int16> {
    using type = std::int16_t;
};
template<>
struct GetType<int32> {
    using type = std::int32_t;
};
template<>
struct GetType<int64> {
    using type = std::int64_t;
};
#if KOTEKAN_FLOAT16
template<>
struct GetType<float16> {
    using type = float16_t;
};
#endif
template<>
struct GetType<float32> {
    using type = float;
};
template<>
struct GetType<float64> {
    using type = double;
};

template<DataType type>
using GetType_t = typename GetType<type>::type;

// Convert a type to a string
std::string type_to_string(DataType type);

// Convert a string to a type
DataType string_to_type(const std::string& type_name);

// Output a type
std::ostream& operator<<(std::ostream& os, DataType type);

} // namespace kotekan

#endif // #ifndef DATATYPE_HPP
