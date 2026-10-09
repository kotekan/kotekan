#ifndef HDF5FILES_HPP
#define HDF5FILES_HPP

#include "DataType.hpp"  // for DataType, float16_t
#include "Telescope.hpp" // for ElementOrder

#include <array>                         // for array
#include <cstddef>                       // for size_t
#include <cstdint>                       // for uint64_t
#include <highfive/H5DataType.hpp>       // for AtomicType, DataType, H5T_NATIVE_FLOAT
#include <highfive/H5File.hpp>           // for File
#include <highfive/H5PropertyList.hpp>   // for DataSetCreateProps
#include <highfive/bits/h5t_wrapper.hpp> // for h5t_copy, h5t_set_ebias, h5t_set_fields, h5t_se...
#include <string>                        // for string
#include <vector>                        // for vector

class CHORDTelescope;

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
 * @brief Whether a dimension is a time axis
 *
 * Kotekan convention: a dimension is a time axis iff its name starts with 'T'
 * ("T", "Tc", "Tbar", "Ttilde", "Thi64", "T8hi128", ...).
 */
inline bool is_time_axis(const std::string& dim_name) {
    return !dim_name.empty() && dim_name[0] == 'T';
}

/**
 * @brief Create a chunked dataset and record its dimension names in an "axis" attribute
 *
 * A 1-D dataset is a single chunk. Otherwise every dimension is chunked whole, except that
 * "time", "freq*" and "prod*"/"el*" dimensions are capped at @p blocksize_t, @p blocksize_f
 * and @p blocksize_p (0 leaves the dimension whole). An existing dataset is left as it is.
 */
void create_dataset_with_axis(HighFive::File& file, const std::string& name,
                              const std::vector<hsize_t>& dims,
                              const std::vector<std::string>& dim_names,
                              const HighFive::DataType& dtype, HighFive::DataSetCreateProps props,
                              std::size_t blocksize_f, std::size_t blocksize_p,
                              std::size_t blocksize_t);

/**
 * @brief Write the per-element input description of a CHORD file into /index_map
 *
 * One row per element of the file's element axis, naming the dish and polarization the
 * element decodes to and copying that dish's entry of the telescope's `dish_inputs` table:
 * /index_map/{dish_idx,pol,grid_x_idx,grid_y_idx,feed_pos_disp_m,coelev_disp_deg,type,label},
 * where the label is the dish label with the polarization name appended (A01X, A01Y). Also
 * writes /index_map/dish_positions_in_grid_coords for every dish of the telescope. The format
 * is documented in docs/sphinx/user/file_formats/n2_vis_hdf5.rst.
 *
 * @param rows         The array element, in @p order, held by each element of the file.
 * @param blocksize_p  Chunk cap for the element dimension (0 = none).
 */
void write_index_map(HighFive::File& file, const CHORDTelescope& telescope,
                     const std::vector<std::uint64_t>& rows, ElementOrder order,
                     std::size_t blocksize_p);

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
