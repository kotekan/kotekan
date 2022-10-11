// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
 * @file   nDiskMultiFormatWriter.hpp
 * @brief  This file declares nDiskMultiFormatWriter
 *         stage, multi-format multiple drives writer
 *
 * @author Mehdi Najafi
 * @date   12 SEP 2022
 *****************************************************/

#ifndef N_DISK_MULTIFORMAT_WRITER_HPP
#define N_DISK_MULTIFORMAT_WRITER_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t, int64_t
#include <string>   // for string
#include <thread>   // for thread
#include <time.h>   // for tm
#include <vector>   // for vector


/**
 * @class nDiskMultiFormatWriter
 * @brief Consumer ``kotekan::Stage`` which writes input data on multiple drives in various formats.
 *
 * This is a consumer which initiates n threads to write to ``n`` disks. Each drive will receive
 * data from every ``n``th buffer, stored within a common-named subfolder. Within each folder
 * the data files will be numbered incrementally across the disks.
 *
 * @par Buffers
 *  @buffer in_buf The kotkean buffer with the data to be written
 *  @buffer_metadata metadata The kotkean metadata to be written
 *
 * @conf num_disks      Int , the number of drives to read from
 * @conf disk_path      String, the path to the mounted drives and address
 * @conf file_format    String, the file format. Currently "raw" and "HDF5" are accepted
 * @conf max_frames_per_file      Int , the number of frames to be written in each file
 * @conf write_to_frame_metadata  Bool, whether to actually save metadata or ignore it
 * @conf write_to_disk  Bool, whether to actually save, alternately operating in dummy mode
 * @conf write_metadata_and_gains  Bool, Default true.  Flag to control if VDIF/ARO style gains
 *                                 and metadata are copied to the acquisition folder.
 *
 * @todo    Add more file formats.
 *
 * An example with n = 2:
 *
 * kotekan::Config Parameters:
 *
 * - num_disk: 2
 * - disk_path: /drives/CHF/{n}/{DATE}T{TIME}Z_aro
 * - file_format: raw
 *
 * This will output data in files like:
 *
 * Drive 0:
 *
 * - /drives/D/0/20220927T162214Z_aro/settings.txt
 * - /drives/D/0/20220927T162214Z_aro/0000000.raw
 * - /drives/D/0/20220927T162214Z_aro/0000002.raw
 * - /drives/D/0/20220927T162214Z_aro/0000004.raw
 *
 * Drive 1:
 *
 * - /drives/D/1/20220927T162214Z_aro/settings.txt
 * - /drives/D/1/20220927T162214Z_aro/0000001.raw
 * - /drives/D/1/20220927T162214Z_aro/0000003.raw
 * - /drives/D/1/20220927T162214Z_aro/0000005.raw
 *
 * The output format is as:
 * 1 int64_t as a number of frames present in the file -> m
 * ... first frame here ...
 * 1 size_t, the size of the metadata structure in bytes -> mbz
 * 1 structure of the byte size: mbz
 * 1 uint32_t, the number of frequencies in each frame -> n
 * 1 size_t, the byte size of each frame data -> fbz = n + n*frequency_data_size
 * 1 array of uint8_t of size fbz
 * ... the next frame until the end or the number of frames reaches m
 *
 * @author Mehdi Najafi
 */
class nDiskMultiFormatWriter : public kotekan::Stage {
public:
    /// Constructor
    nDiskMultiFormatWriter(kotekan::Config& config, const std::string& unique_name,
                           kotekan::bufferContainer& buffer_container);

    /// Destructor, currently does nothing
    virtual ~nDiskMultiFormatWriter(){};

    /// Creates n safe instances of the file_read_thread thread
    void main_thread() override;

private:
    /// The kotekan buffer object the stage is consuming from
    struct Buffer* buf;

    /// the file format and file extension
    std::string file_format, file_extension;

    /// A holder for the config parameter num_disks
    uint32_t num_disks;

    /// Which disk in the array is currently being written to
    uint32_t disk_id;

    /// maximum number of frames expected in each file
    int64_t max_frames_per_file;

    /// the number of frequencies expected in each incoming frame
    uint32_t num_freq_per_output_frame;

    /// raw file writer thread function
    void raw_file_write_thread(int disk_id);

    /// hdf5 file writer thread function
    void hdf5_file_write_thread(int disk_id);

    /// array of file writer threads
    std::vector<std::thread> file_thread_handles;

    /// The subfolder name where the files will be stored
    std::string dataset_name;

    /// A path on an n-disk array used to put the files there
    std::string disk_path;

    /// Boolean config parameter to enable or disable file output
    bool write_to_disk;

    /// Flag to enable or disable writing out the metadata and gains
    bool write_metadata_and_gains;

    /// The date/time of that this stage started running
    std::tm invocation_time;

    /// make the dataset folder name out of the given format in disk_path
    std::string get_dataset_folder_name(const int n);

    /// provide an extension based on the given file_format
    std::string get_file_extension();

    /// make the timestamp and return it as a string
    std::string get_dataset_timestamp();

    /// Function to make subdirectories dataset_name on each disk in the disk set
    void make_folders();

    /// replace all occurrences of a substring by another string
    void string_replace_all(std::string& str, const std::string& from, const std::string& to);

    /// Function to write relevant config parameters to a settings.txt file
    void save_metadata();

    /// Function to backup the FPGA gain file alongside the data
    void copy_gains(const std::string& gain_file_dir, const std::string& gain_file_name);
};

#endif // N_DISK_MULTIFORMAT_WRITER_HPP
