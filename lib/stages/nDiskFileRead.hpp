/**
 * @file
 * @brief A stage to read VDIF files from multiple drives.
 *  - nDiskFileRead : public kotekan::Stage
 */

#ifndef N_DISK_FILE_READ_H
#define N_DISK_FILE_READ_H

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string, basic_string
#include <thread>   // for thread
#include <vector>   // for vector

/**
 * @class nDiskFileRead
 * @brief Producer ``kotekan::Stage`` which reads VDIF data from multiple drives into a
 * ``Buffer``
 *
 * This is a producer which initiates n threads to read from n disks. Each disk must contain data in
 * the same folders as specified in the kotekan config file. Within each folder the data files must
 * be numbered incrementally across the disks. Since the file format is the most important aspect of
 * this stage, a worked example for a set of 3 disks is shown below.
 *
 * @par Buffers
 * @buffer out_buf The kotkean buffer to hold the data read from the drives
 * 	@buffer_format Array of unsigned char, just copies the file.
 * 	@buffer_metadata none
 *
 * @conf num_disks            Int. Number of drives to read from.
 * @conf disk_base            String. Path to mounted drives (e.g. `/drives/`).
 * @conf disk_set             String. Disk set prefix (e.g. `D`).
 * @conf capture              String. Subfolder of current data set.
 * @conf starting_file_index  Int. Starting file index offset.
 *
 * @warning 	Not getting the file format correct will usually result in a segmentation fault. It
 * can be hard to figure out what is happening, so be extra cautious.
 *
 * @todo	Add rest server commands.
 *
 * Worked Example with n = 3:
 *
 * kotekan::Config Parameters:
 *
 * - num_disk: 3
 * - disk_base: /drives/
 * - disk_set: /D/
 * - capture: 20170805T155218Z_aro_vdif
 * - starting_index: 0
 *
 * What the file paths should look like:
 *
 * Drive 0:
 *
 * - /drives/D/0/20170805T155218Z_aro_vdif/0000000.vdif
 * - /drives/D/0/20170805T155218Z_aro_vdif/0000003.vdif
 * - /drives/D/0/20170805T155218Z_aro_vdif/0000006.vdif
 *
 * Drive 1:
 *
 * - /drives/D/1/20170805T155218Z_aro_vdif/0000001.vdif
 * - /drives/D/1/20170805T155218Z_aro_vdif/0000004.vdif
 * - /drives/D/1/20170805T155218Z_aro_vdif/0000007.vdif
 *
 * Drive 2:
 *
 * - /drives/D/2/20170805T155218Z_aro_vdif/0000002.vdif
 * - /drives/D/2/20170805T155218Z_aro_vdif/0000005.vdif
 * - /drives/D/2/20170805T155218Z_aro_vdif/0000008.vdif
 *
 * @par Example
 * @code
 * n_disk_file_read:
 *   kotekan_stage: nDiskFileRead
 *   out_buf: vdif_in
 *   num_disks: 3
 *   disk_base: /drives/
 *   disk_set: D
 *   capture: 20170805T155218Z_aro_vdif
 *   starting_file_index: 0
 * @endcode
 *
 * @author Jacob Taylor
 */
class nDiskFileRead : public kotekan::Stage {
public:
    /// Constructor
    nDiskFileRead(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_containter);

    /// Destructor, currently does nothing
    ~nDiskFileRead() override{};

    /**
     * Entrance point for n threads.
     * Reads files from a given drive in order
     * and places the file contents into a kotekan buffer.
     *
     * @param disk_id   Tells the function which disk to read off of.
     *                  The function will read off of the disk indicated by
     *                  disk_id.
     */
    void file_read_thread(int disk_id);

    /// Creates n safe instances of the file_read_thread thread
    void main_thread() override;

private:
    /// The kotekan buffer object the stage is producing for
    Buffer* buf;
    /// Vector to hold the thread handles
    std::vector<std::thread> file_thread_handles;
    /// A holder for the config parameter num_disks
    uint32_t num_disks;
    /// A holder for the config parameter starting_file_index
    uint32_t starting_index;
    /// A holder for the config parameter disk_base
    std::string disk_base;
    /// A holder for the config parameter disk_set
    std::string disk_set;
    /// A holder for the config parameter capture
    std::string capture;
};

#endif
