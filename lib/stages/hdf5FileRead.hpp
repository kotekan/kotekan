/**
 * @file
 * @brief Reads frames from an HDF5 file into a kotekan buffer.
 * - hdf5FileRead : public kotekan::Stage
 */

#ifndef HDF5_FILE_READ_HPP
#define HDF5_FILE_READ_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

/**
 * @class hdf5FileRead
 * @brief Reads frames from an HDF5 file into a kotekan buffer.
 *
 * @par Buffers
 * @buffer out_buf Buffer to write frames into.
 *     @buffer_format Matches file contents (Vis/HFB/N2/CHORD)
 *     @buffer_metadata Matches file contents (VisMetadata/HFBMetadata/N2Metadata/chordMetadata)
 *
 * @conf out_buf              String. Output buffer.
 * @conf input_dir            String. Directory containing the file.
 * @conf file_name            String. File name stem (without numeric index/extension).
 * @conf prefix_hostname      Bool, default true. Prefix file name with hostname.
 * @conf prefix_host_rank     Bool, default false. Prefix with host pool rank.
 * @conf frequency_pool_rank  Int, default 0. Rank index for pool layouts.
 * @conf frequency_pool_size  Int, default 1. Pool size for frequency distribution.
 * @conf do_once              Bool, default false. Stop after first frame if true.
 * @conf read_mode            String, default "auto". Optional override for file type (vis/hfb/n2).
 * @conf max_frames           Int, default -1. Stop after this many frames (-1 = unlimited).
 *
 * @par Metrics
 * @metric kotekan_hdf5fileread_read_time_seconds  Time to read last frame.
 *
 * @par Example
 * @code
 * hdf5FileRead:
 *   out_buf: vis_in
 *   input_dir: /data/h5
 *   file_name: vis_dump
 *   prefix_hostname: true
 *   prefix_host_rank: false
 *   frequency_pool_rank: 0
 *   frequency_pool_size: 1
 *   do_once: false
 * @endcode
 */

#endif
