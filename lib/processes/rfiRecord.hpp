/**
 * @file rfiRecorder.hpp
 * @brief Contains RFI data recorder for SK estimates in kotekan.
 *  - rfiRecord : public KotekanProcess
 */

#ifndef RFI_RECORD_H
#define RFI_RECORD_H

#include "powerStreamUtil.hpp"
#include <sys/socket.h>
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "chimeMetadata.h"


/**
 * @author Jacob Taylor
 */


class rfiRecord : public KotekanProcess {
public:
    //Constructor, intializes config variables via apply_config
    rfiRecord(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);

    //Deconstructor, cleans up / does nothing
    virtual ~rfiRecord();

    //Primary loop, reads buffer and sends out UDP stream
    void main_thread();

    void save_meta_data(uint16_t streamID, int64_t firstSeqNum);
    //Intializes config variables
    virtual void apply_config(uint64_t fpga_seq);

private:
    /// Kotekan buffer containing kurtosis estimates
    struct Buffer *rfi_buf;

    //General Config Parameters
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Total number of frequencies (1024)
    uint32_t _num_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;

    //RFI config parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t  _sk_step;
    /// Flag for element summation in kurtosis estimation process
    bool COMBINED;
    /// Number of frames to average per UDP packet
    uint32_t frames_per_packet;

    //Process specific config parameters
    /// The total number of links processed by gpu
    uint32_t total_links;
    string write_to;
    char time_dir[50];
    bool write_to_disk;
};

#endif
