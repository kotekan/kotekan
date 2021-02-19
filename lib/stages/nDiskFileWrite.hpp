/**
 * @file
 * @brief A stage to read VDIF files from multiple drives.
 *  - nDiskFileWrite : public kotekan::Stage
 */

#ifndef N_DISK_FILE_WRITE_H
#define N_DISK_FILE_WRITE_H
#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <thread>   // for thread
#include <vector>   // for vector


/**
 * @class nDiskFileWrite
 * @brief Consumer ``kotekan::Stage`` which writes VDIF-formatted input  data on multiple
 * drives.
 *
 * This is a consumer which initiates n threads to write to ``n`` disks. Each drive will receive
 * data from every
 * ``n``th buffer, stored within a common-named subfolder. Within each folder the data files will be
 * numbered incrementally across the disks.
 *
 * @par Buffers
 * @buffer in_buf The kotkean buffer holing the data to be written
 *  @buffer_format Array of VDIF frames.
 *  @buffer_metadata none
 *
 * @conf num_disks      Int , the number of drives to read from
 * @conf disk_base      String, the path to the mounted drives
 * @conf disk_set       String, the disk name.
 * @conf file_ext       String, Default 'vdif', the extenstion of the output file.
 * @conf write_to_disk  Bool, whether to actually save, alternately operating in dummy mode
 * @conf instrument_name String, used in filenames and stored to metadata text file.
 * @conf write_metadata_and_gains  Bool, Default true.  Flag to control if VDIF/ARO style gains
 *                                 and metadata are copied to the acquisition folder.
 * @conf print_lost_sample_number Bool, Default true. Flag to control if the number of lost
 *                                 sample is get printed in the std out.
 * @todo    Make more general, to support more than just ICEboard-generated data.
 *
 * Worked Example with n = 3:
 *
 * kotekan::Config Parameters:
 *
 * - num_disk: 3
 * - disk_base: /drives/
 * - disk_set: /D/
 * - instrument_name: aro
 *
 * This will output data in files like:
 *
 * Drive 0:
 *
 * - /drives/D/0/20170805T155218Z_aro_vdif/settings.txt
 * - /drives/D/0/20170805T155218Z_aro_vdif/0000000.vdif
 * - /drives/D/0/20170805T155218Z_aro_vdif/0000003.vdif
 * - /drives/D/0/20170805T155218Z_aro_vdif/0000006.vdif
 *
 * Drive 1:
 *
 * - /drives/D/1/20170805T155218Z_aro_vdif/settings.txt
 * - /drives/D/1/20170805T155218Z_aro_vdif/0000001.vdif
 * - /drives/D/1/20170805T155218Z_aro_vdif/0000004.vdif
 * - /drives/D/1/20170805T155218Z_aro_vdif/0000007.vdif
 *
 * Drive 2:
 *
 * - /drives/D/2/20170805T155218Z_aro_vdif/settings.txt
 * - /drives/D/2/20170805T155218Z_aro_vdif/0000002.vdif
 * - /drives/D/2/20170805T155218Z_aro_vdif/0000005.vdif
 * - /drives/D/2/20170805T155218Z_aro_vdif/0000008.vdif
 *
 * @author Andre Renard
 */
class nDiskFileWrite : public kotekan::Stage {
public:
    /// Constructor
    nDiskFileWrite(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_containter);

    /// Destructor, currently does nothing
    virtual ~nDiskFileWrite();

    /// Creates n safe instances of the file_read_thread thread
    void main_thread() override;

private:
    /// The kotekan buffer object the stage is consuming from
    struct Buffer* buf;

    /// Which disk in the array is currently being written to
    uint32_t disk_id;
    /// A holder for the config parameter num_disks
    uint32_t num_disks;

    void file_write_thread(int disk_id);
    std::vector<std::thread> file_thread_handles;

    /// The subfolder name where the files will be stored
    std::string dataset_name;
    /// A holder for the config parameter disk_base, where the drive sets are mounted
    std::string disk_base;
    /// A holder for the config parameter disk_set, where to write files
    std::string disk_set;
    /// The out put file extenstion
    std::string file_ext;
    /// Boolean config parameter to enable or disable file output
    bool write_to_disk;

    /// Flag to enable or disable writing out the metadata and gains
    bool write_metadata_and_gains;

    /// Flag to enable or disable print out the lost sample number in stdout
    bool print_lost_sample_number;

    /// Function to make subdirectories dataset_name on each disk in the disk set
    void mk_dataset_dir();

    /// Function to write relevant config parameters to a settings.txt file alongside the VDIF data
    void save_meta_data(char* timestr);

    /// Function to back up the FPGA gain file alongside the VDIF data
    void copy_gains(const std::string& gain_file_dir, const std::string& gain_file_name);

    /// A holder for the config parameter instrument name, used in VDIF header
    std::string instrument_name;
};

#endif
