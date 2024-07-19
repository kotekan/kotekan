#ifndef GDALFILES_HPP
#define GDALFILES_HPP

#include <array>
#include <chordMetadata.hpp>
#include <complex>
#include <gdal.h>
#include <type_traits>
#include <vector>

namespace gdal {

// The version number of the CHORD metadata format in GDAL files.
// This uses semver: A file with version X.Y can be read by software
// that understands X.Z even when Y > Z.
constexpr std::array<int, 2> chord_metadata_version{1, 0};

GDALDataType chord2gdal(const chordDataType type);

template<typename T>
struct gdal_datatype;
template<>
struct gdal_datatype<unsigned char> : std::integral_constant<GDALDataType, GDT_Byte> {};
template<>
struct gdal_datatype<unsigned short> : std::integral_constant<GDALDataType, GDT_UInt16> {};
template<>
struct gdal_datatype<unsigned int> : std::integral_constant<GDALDataType, GDT_UInt32> {};
template<>
struct gdal_datatype<unsigned long>
    : std::integral_constant<GDALDataType, (sizeof(unsigned long) == 4 ? GDT_UInt32 : GDT_UInt64)> {
};
template<>
struct gdal_datatype<unsigned long long> : std::integral_constant<GDALDataType, GDT_UInt64> {};
template<>
struct gdal_datatype<signed char> : std::integral_constant<GDALDataType, GDT_Int8> {};
template<>
struct gdal_datatype<short> : std::integral_constant<GDALDataType, GDT_Int16> {};
template<>
struct gdal_datatype<int> : std::integral_constant<GDALDataType, GDT_Int32> {};
template<>
struct gdal_datatype<long>
    : std::integral_constant<GDALDataType, (sizeof(unsigned long) == 4 ? GDT_Int32 : GDT_Int64)> {};
template<>
struct gdal_datatype<long long> : std::integral_constant<GDALDataType, GDT_Int64> {};
template<>
struct gdal_datatype<char>
    : std::integral_constant<GDALDataType, (std::is_signed_v<char> ? GDT_Int8 : GDT_Byte)> {};
#if KOTEKAN_FLOAT16
template<>
struct gdal_datatype<float16_t> : std::integral_constant<GDALDataType, GDT_UInt16> {};
#endif
template<>
struct gdal_datatype<float> : std::integral_constant<GDALDataType, GDT_Float32> {};
template<>
struct gdal_datatype<double> : std::integral_constant<GDALDataType, GDT_Float64> {};
template<>
struct gdal_datatype<std::complex<short>> : std::integral_constant<GDALDataType, GDT_CInt16> {};
template<>
struct gdal_datatype<std::complex<int>> : std::integral_constant<GDALDataType, GDT_CInt32> {};
#if KOTEKAN_FLOAT16
template<>
struct gdal_datatype<std::complex<float16_t>> : std::integral_constant<GDALDataType, GDT_UInt32> {};
#endif
template<>
struct gdal_datatype<std::complex<float>> : std::integral_constant<GDALDataType, GDT_CFloat32> {};
template<>
struct gdal_datatype<std::complex<double>> : std::integral_constant<GDALDataType, GDT_CFloat64> {};

template<class T>
inline constexpr GDALDataType gdal_datatype_v = gdal_datatype<T>::value;

template<typename T>
constexpr GDALDataType get_gdal_datatype(T) {
    return gdal_datatype_v<std::remove_reference_t<std::remove_cv_t<T>>>;
}

std::vector<const char*> convert_to_cstring_list(const std::vector<std::string>& strings);

} // namespace gdal

#endif // #ifndef GDALFILES_HPP
