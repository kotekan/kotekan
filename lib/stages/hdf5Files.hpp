#ifndef HDF5FILES_HPP
#define HDF5FILES_HPP

#include "DataType.hpp" // for DataType, float16_t

#include "json.hpp" // for json

#include <array>                         // for array
#include <chrono>                        // for system_clock
#include <cstddef>                       // for size_t
#include <highfive/H5DataType.hpp>       // for AtomicType, DataType, H5T_NATIVE_FLOAT
#include <highfive/H5File.hpp>           // for File
#include <highfive/bits/h5t_wrapper.hpp> // for h5t_copy, h5t_set_ebias, h5t_set_fields, h5t_se...
#include <map>                           // for map
#include <memory>                        // for shared_ptr
#include <string>                        // for string
#include <vector>                        // for vector

namespace hdf5 {

// The version number of the CHORD metadata format in HDF5 files.
// This uses semver: A file with version X.Y can be read by software
// that understands X.Z even when Y > Z.
constexpr std::array<int, 2> chord_metadata_version{2, 0};

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

/**
 * @brief The common part of the file names used by hdf5FileWrite and hdf5FileRead:
 *        "<dir>/[<hostname>_][x<rank:04d>_]<file_name>"
 *
 * The caller appends the suffix, i.e. ".<frame:08d>.h5" for per-frame files and ".h5" for a
 * single file.
 */
std::string file_path_prefix(const std::string& dir, const std::string& file_name,
                             bool prefix_hostname, bool prefix_host_rank, int host_pool_rank);

/**
 * @brief An acquisition directory: "<base_dir>/acq_YYYYMMDD_HHMMSS_NNNNNNNNN" for time @p t (UTC)
 */
std::string acq_dir_path(const std::string& base_dir, std::chrono::system_clock::time_point t);

/**
 * @brief The start time of this run, which names its acquisition directories
 *
 * Holders share one time. After the last holder is released the next call takes a new one,
 * so the stages of one pipeline share it and a pipeline started after /stop gets its own.
 */
std::shared_ptr<const std::chrono::system_clock::time_point> acq_run_time();

/**
 * @brief Whether a dimension is a time axis
 *
 * Kotekan convention: a dimension is a time axis iff its name starts with 'T'
 * ("T", "Tc", "Tbar", "Ttilde", "Thi64", "T8hi128", ...).
 */
inline bool is_time_axis(const std::string& dim_name) {
    return !dim_name.empty() && dim_name[0] == 'T';
}

/// The group hdf5FileWrite stores the per-frame metadata in (record_per_frame_metadata).
constexpr const char* frame_metadata_group = "frame_metadata";

/**
 * @brief One value of a frame's metadata as it is stored per frame: a scalar or a
 *        rectangular array of booleans, integers, floats or strings.
 */
struct FrameMetadataColumn {
    enum Kind { Bool, Int, UInt, Float, String };
    Kind kind = Int;
    /// The shape of one frame's value, empty for a scalar.
    std::vector<std::size_t> shape;
    /// The elements in row-major order.
    std::vector<nlohmann::json> values;
    /// For strings, the length of the longest element, or the stored length of a file's column.
    std::size_t width = 0;
};

/// The columns by dataset path inside the group: "KEY", or "KEY/SUBKEY" for a nested object.
using FrameMetadataColumns = std::map<std::string, FrameMetadataColumn>;

/**
 * @brief Flatten the json of one frame's metadata into columns.
 *
 * Throws std::runtime_error for a value that cannot be a column: null, an empty array or
 * object, a ragged array, or an array of mixed kinds.
 */
FrameMetadataColumns frame_metadata_columns(const nlohmann::json& metadata);

/**
 * @brief Create the empty datasets of @p columns in @p file.
 *
 * @return The columns of the file: @p columns with the stored string lengths.
 */
FrameMetadataColumns create_frame_metadata(HighFive::File& file, FrameMetadataColumns columns,
                                           bool use_compression);

/// Whether @p frame has the keys, kinds and shapes of @p file_columns, and strings that fit.
bool frame_metadata_fits(const FrameMetadataColumns& file_columns,
                         const FrameMetadataColumns& frame);

/// Append @p frame, which must fit, as one row of every column.
void append_frame_metadata(HighFive::File& file, const FrameMetadataColumns& file_columns,
                           const FrameMetadataColumns& frame);

/**
 * @brief Read the per-frame metadata of @p file back, one json object per frame.
 *
 * If the columns disagree on the number of frames, as after a writer was killed while it
 * appended one, only the frames that every column has are returned. Throws
 * std::runtime_error if a column holds a type that create_frame_metadata does not write.
 */
std::vector<nlohmann::json> read_frame_metadata(const HighFive::File& file);

} // namespace hdf5

// HighFive owns this template; its definitions are outside the API documentation.
/// @cond
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
/// @endcond

#endif // #ifndef HDF5FILES_HPP
