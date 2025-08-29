#ifndef HDF5FILES_HPP
#define HDF5FILES_HPP

#include <DataType.hpp>
#include <array>
#include <highfive/highfive.hpp>

namespace hdf5 {

// The version number of the CHORD metadata format in HDF5 files.
// This uses semver: A file with version X.Y can be read by software
// that understands X.Z even when Y > Z.
constexpr std::array<int, 2> chord_metadata_version{1, 0};

// Bitshuffle flags
constexpr unsigned int H5Z_BLOSC = 32001;      // blosc filter id
constexpr unsigned int H5Z_BITSHUFFLE = 32008; // bitshuffle filter id
constexpr unsigned int H5Z_BLOSC2 = 32026;     // bitshuffle filter id

constexpr unsigned int BLOSC_SHUFFLE_NONE = 0;
constexpr unsigned int BLOSC_SHUFFLE_BYTE = 1;
constexpr unsigned int BLOSC_SHUFFLE_BIT = 2;
constexpr unsigned int BLOSC_COMPRESS_BLOSCLZ = 0;
constexpr unsigned int BLOSC_COMPRESS_LZ4 = 1;
constexpr unsigned int BLOSC_COMPRESS_LZ4HC = 2;
constexpr unsigned int BLOSC_COMPRESS_SNAPPY = 3;
constexpr unsigned int BLOSC_COMPRESS_ZLIB = 4;
constexpr unsigned int BLOSC_COMPRESS_ZSTD = 5;

constexpr unsigned int BITSHUFFLE_BLOCKSIZE_AUTO = 0; // let bitshuffle choose block size
constexpr unsigned int BITSHUFFLE_COMPRESS_NONE = 0;
constexpr unsigned int BITSHUFFLE_COMPRESS_LZ4 = 2;
constexpr unsigned int BITSHUFFLE_COMPRESS_ZSTD = 3;

HighFive::DataType chord2hdf5(const kotekan::DataType type);

} // namespace hdf5

namespace HighFive {
template<>
inline HighFive::AtomicType<float16_t>::AtomicType() {
    _hid = detail::h5t_copy(H5T_NATIVE_FLOAT);
    // Sign position, exponent position, exponent size, mantissa position, mantissa size
    detail::h5t_set_fields(_hid, 15, 10, 5, 0, 10);
    // Total datatype size (in bytes)
    detail::h5t_set_size(_hid, 2);
    // Floating point exponent bias
    detail::h5t_set_ebias(_hid, 15);
}
} // namespace HighFive

#endif // #ifndef HDF5FILES_HPP
