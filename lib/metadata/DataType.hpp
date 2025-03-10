#ifndef DATATYPE_HPP
#define DATATYPE_HPP

#include <cstdint>
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

namespace chord {

// This enum lets us talk about the various datatypes we're using.
enum DataType {
    unknown_type,
    uint1x8, // 8 bools (packed into a type)
    uint4p4, // 2 unsigned 4-bit integers (packed into a byte)
    uint8,
    uint16,
    uint32,
    uint64,
    int4p4,      // 2 signed 4-bit integers (packed into a byte)
    int4p4chime, // offset-encoded (stored is value + 8), low and high values swapped
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
};

// Convert a type to a string
std::string type_to_string(DataType type);

// Output a type
std::ostream& operator<<(std::ostream& os, DataType type);

// Number of bits (not bytes!) in a type. For packed types, say how many bits there are in each
// element.
constexpr std::size_t type_bits(DataType type) {
    switch (type) {
        case uint1x8:
            return 1;
        case uint4p4:
            return 4;
        case uint8:
            return 8;
        case uint16:
            return 16;
        case uint32:
            return 32;
        case uint64:
            return 64;
        case int4p4:
            return 4;
        case int4p4chime:
            return 4;
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
            return 0;
    }
}

// Find an unsigned int with that many bits
constexpr DataType uint_from_bits(std::size_t bits) {
    switch (bits) {
        case 1:
            return uint1x8;
        case 4:
            return uint4p4;
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
constexpr DataType int_from_bits(std::size_t bits) {
    switch (bits) {
        case 4:
            return int4p4;
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
constexpr DataType float_from_bits(std::size_t bits) {
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
    : std::integral_constant<DataType, uint_from_bits(8 * sizeof(unsigned char))> {};
template<>
struct GetDataType<unsigned short>
    : std::integral_constant<DataType, uint_from_bits(8 * sizeof(unsigned short))> {};
template<>
struct GetDataType<unsigned int>
    : std::integral_constant<DataType, uint_from_bits(8 * sizeof(unsigned int))> {};
template<>
struct GetDataType<unsigned long>
    : std::integral_constant<DataType, uint_from_bits(8 * sizeof(unsigned long))> {};
template<>
struct GetDataType<unsigned long long>
    : std::integral_constant<DataType, uint_from_bits(8 * sizeof(unsigned long long))> {};
template<>
struct GetDataType<signed char>
    : std::integral_constant<DataType, int_from_bits(8 * sizeof(signed char))> {};
// We omit char because we don't know whether it's signed or unsigned
template<>
struct GetDataType<short> : std::integral_constant<DataType, int_from_bits(8 * sizeof(short))> {};
template<>
struct GetDataType<int> : std::integral_constant<DataType, int_from_bits(8 * sizeof(int))> {};
template<>
struct GetDataType<long> : std::integral_constant<DataType, int_from_bits(8 * sizeof(long))> {};
template<>
struct GetDataType<long long>
    : std::integral_constant<DataType, int_from_bits(8 * sizeof(long long))> {};
#if KOTEKAN_FLOAT16
template<>
struct GetDataType<float16_t>
    : std::integral_constant<DataType, float_from_bits(8 * sizeof(float16_t))> {};
#endif
template<>
struct GetDataType<float> : std::integral_constant<DataType, float_from_bits(8 * sizeof(float))> {};
template<>
struct GetDataType<double> : std::integral_constant<DataType, float_from_bits(8 * sizeof(double))> {
};
template<>
struct GetDataType<long double>
    : std::integral_constant<DataType, float_from_bits(8 * sizeof(long double))> {};

// Use e.g. as `DataType double_type = GetDataType_v<double>`
template<typename T>
constexpr DataType GetDataType_v = GetDataType<T>::value;

} // namespace chord

#endif // #ifndef DATATYPE_HPP
