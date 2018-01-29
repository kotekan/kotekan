/**
 *  * @file nDiskFileRead.hpp
 *   * @brief A process to read VDIF files from multiple drives.
 *    *  - nDiskFileRead : public KotekanProcess
 *     */

#ifndef N_DISK_FILE_READ_H
#define N_DISK_FILE_READ_H

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "vdif_functions.h"

/**
 * @class nDiskFileRead
 * @brief Producer ``KotekanProcess`` which reads VDIF data from multiple drives into a ``Buffer``
 *
 * This is a producer which initiates n threads to read from n disks. Each disk must contain data in the
 * same folders as specified in the kotekan config file. Within each folder the data files must be numbered
 * incrementally across the disks. Since the file format is the most important aspect of this process, a worked
 * example for a set of 3 disks is shown below.
 *
 * @par Buffers
 * @buffer out_buf The kotkean buffer to hold the data read from the drives
 * 	@buffer_format Array of unsigned char, just copies the file. 
 * 	@buffer_metadata none
 *
 * @conf num_disks		Int , the number of drives to read from (Example: 10)
 * @conf disk_base		String, the path to the mounted drives (Example: '/drives/')
 * @conf disk_set		String, the disk name (Example: 'D')
 * @conf capture		String, the subfolder of the current data set (Example: 20170805T155218Z_aro_vdif)
 * @conf starting_file_index 	Int, an offset for where to start in the data set (Example: 10232)
 *
 * @warning 	Not getting the file format correct will usually result in a segmentation fault. It can be hard to
 * 		figure out what is happening, so be extra cautious.
 *
 * @todo	Add rest server commands.
 *
 * Worked Example with n = 3:
 * 
 * Config Parameters:
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
 * @author Jacob Taylor
 */

class nDiskFileRead : public KotekanProcess {
public:
    //Constructor, calls apply_config to intialize parameters
    nDiskFileRead(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_containter);
    //Destructor, currently does nothing 
    ~nDiskFileRead();

    /**
     * Entrance point for n threads. 
     * Reads files from a given drive in order
     * and places the file contents into a kotekan buffer.
     * @param disk_id 	Tells the function which disk to read off of. 
     * 			The function will read off of the disk indicated by
     * 			disk_id.
     */
    void file_read_thread(int disk_id);

    //Applies the config parameters
    void apply_config(uint64_t fpga_seq) override;
    //Creates n safe threads of file_read_thre
    void main_thread() override;
private:
    //The kotekan buffer object the processes is producing for
    struct Buffer *buf;
    //Vector to hold the thread handles
    std::vector<std::thread> file_thread_handles;
    //A holder for the config parameter num_disks
    uint32_t num_disks;
    //A holder for the config parameter starting_file_index
    uint32_t starting_index;
    //A holder for the config parameter disk_base
    string disk_base;
    //A holder for the config parameter disk_set
    string disk_set;
    //A holder for the config parameter capture
    string capture; 
};

#endif
