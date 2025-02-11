#ifndef DATATYPE_HPP
#define DATATYPE_HPP

#include <cstdint>
#include <iostream>
#include <type_traits>

enum DataType {
    unknown_type,
    uint4p4,
    uint8,
    uint16,
    uint32,
    uint64,
    int4p4,
    int4p4chime, // offset-encoded (stored is value + 8), low and high values swapped
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
};

std::ostream& operator<<(std::ostream& os, DataType type);

constexpr std::size_t type_bits(DataType type) {
    switch (type) {
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

constexpr DataType uint_from_bits(std::size_t bits) {
    switch (bits) {
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
template<>
struct GetDataType<short> : std::integral_constant<DataType, int_from_bits(8 * sizeof(short))> {};
template<>
struct GetDataType<int> : std::integral_constant<DataType, int_from_bits(8 * sizeof(int))> {};
template<>
struct GetDataType<long> : std::integral_constant<DataType, int_from_bits(8 * sizeof(long))> {};
template<>
struct GetDataType<long long>
    : std::integral_constant<DataType, int_from_bits(8 * sizeof(long long))> {};
template<>
struct GetDataType<float> : std::integral_constant<DataType, float_from_bits(8 * sizeof(float))> {};
template<>
struct GetDataType<double> : std::integral_constant<DataType, float_from_bits(8 * sizeof(double))> {
};
template<>
struct GetDataType<long double>
    : std::integral_constant<DataType, float_from_bits(8 * sizeof(long double))> {};
template<typename T>
constexpr DataType GetDataType_v = GetDataType<T>::value;

#endif // #ifndef DATATYPE_HPP
