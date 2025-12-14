/**
 * @file
 * @brief Stream a buffer to HDF5 files on disk.
 * - hdf5FileWrite : public kotekan::Stage
 */

#ifndef HDF5_FILE_WRITE_HPP
#define HDF5_FILE_WRITE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

/**
 * @class hdf5FileWrite
 * @brief Stream a buffer to disk as HDF5 files.
 *
 * Writes each frame to an HDF5 file under the base name and directory, with optional
 * hostname/rank prefixes. Stops after a configured number of frames/writers if set.
 *
 * @par Buffers:
 * @buffer in_buf Buffer to write to disk.
 *     @buffer_format Any
 *     @buffer_metadata Any (Vis/N2/HFB/CHORD)
 *
 * @conf in_buf                String. Input buffer.
 * @conf base_dir              String. Directory to write into.
 * @conf file_name             String. Base filename stem.
 * @conf prefix_hostname       Bool. Default true. Prefix with hostname.
 * @conf prefix_host_rank      Bool. Default false. Prefix with host pool rank.
 * @conf frequency_pool_rank   Int. Default 0. Rank index for pool layouts.
 * @conf frequency_pool_size   Int. Default 1. Pool size for frequency distribution.
 * @conf max_frames            Int. Default -1. Stop after this many frames (per writer), -1 = unlimited.
 * @conf skip_writing          Bool. Default false. If true, skip actual file writes.
 * @conf exit_after_n_frames   Int. Deprecated alias for max_frames (may be present in code).
 * @conf exit_with_n_writers   Int. Default 0. Exit after this many writers finish.
 * @conf compression           String, default "none". HDF5 compression filter.
 * @conf compression_level     Int, default 0. Compression level for filter.
 * @conf use_bitshuffle        Bool, default false. Enable bitshuffle filter.
 * @conf chunk_size            Array<Int>, optional. Dataset chunk shape (freq, prod, time).
 *
 * @par Metrics
 * @metric kotekan_hdf5filewrite_write_time_seconds  The write time for the last frame.
 *
 * @par Example
 * @code
 * hdf5FileWrite:
 *   in_buf: vis_out
 *   base_dir: /data/h5_out
 *   file_name: vis
 *   prefix_hostname: true
 *   prefix_host_rank: false
 *   frequency_pool_rank: 0
 *   frequency_pool_size: 1
 *   max_frames: -1
 *   skip_writing: false
 * @endcode
 */

#endif
