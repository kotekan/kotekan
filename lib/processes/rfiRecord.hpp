/**
 * @file rfiRecorder.hpp
 * @brief Contains RFI data recorder for SK estimates in kotekan.
 *  - rfiRecord : public KotekanProcess
 */

#ifndef RFI_RECORD_H
#define RFI_RECORD_H

#include <sys/socket.h>
#include "powerStreamUtil.hpp"
#include "Config.hpp"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "restServer.hpp"
#include "chimeMetadata.h"

/*
 * @class rfiRecord
 * @brief Consumer ``KotekanProcess`` which consumes and record a buffer filled with spectral kurtosis estimates.
 *
 * This process reads spectral kurtosis estimate from the GPU/CPU and records them to file. The process will
 * create sub driectories for each stream and data acquisition session. The process will record a new file
 * every 1024 frames. An info file will be availiable in each directory indicating all the dataset parameters.
 *
 * @par Buffers
 * @buffer rfi_in       The kotekan buffer containing spectral kurtosis estimates to be read by the process.
 *      @buffer_format  Array of @c floats
 *      @buffer_metadata chimeMetadata
 *
 * @par REST Endpoints
 * @endpoint    /rfi_record ``POST`` Updates write locaton, meteadat info, toggles writting to disk.
 *                                   Note, calling this endpoint will start a new acquisition.
 *              requires json values      "write_to", "write_to_disk"
 *              update config             "write_to", "write_to_disk"
 *
 * @conf   num_elements         Int . Number of elements.
 * @conf   num_local_freq       Int . Number of local freq.
 * @conf   num_total_freq       Int (default 1024). Number of total freq.
 * @conf   samples_per_data_set Int . Number of time samples in a data set.
 * @conf   sk_step              Int (default 256). Length of time integration in SK estimate.
 * @conf   rfi_combined         Bool (default true). Whether or not the kurtosis measurements include an input sum.
 * @conf   total_links          Int (default 1). Number of FPGA links per buffer.
 * @conf   write_to             String . Path to directory where the process will record data.
 * @conf   write_to_dsk         Bool (default false). Whether or not the proccess will wirte to disk.
 *
 * @author Jacob Taylor
 */
class rfiRecord : public KotekanProcess {
public:
    //Constructor
    rfiRecord(Config& config,
                       const string& unique_name,
                       bufferContainer& buffer_container);
    //Deconstructor, cleans up / does nothing
    virtual ~rfiRecord();
    //Primary loop, reads buffer and sends out UDP stream
    void main_thread();
    //Callback function called by rest server
    void rest_callback(connectionInstance& conn, json& json_request);

private:
    /*
     * @brief  Creates acquisition folders and saves metadata file
     * @param streamID    The unique id of the current stream
     * @param firstSeqNum The first sequence number received by kotekan
     * @param tv          Timeval of when the first packet was received
     * @param ts          Timespec of gps time of when first packet was received
     */
    void save_meta_data(uint16_t streamID, int64_t firstSeqNum, timeval tv, timespec ts);
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
    bool _rfi_combined;
    //Process specific config parameters
    /// The total number of links processed by gpu
    uint32_t _total_links;
    /// The current file index
    uint32_t file_num;
    /// Where to record the RFI data
    string _write_to;
    /// Holder for time-code directory name
    char time_dir[50];
    /// Whether or not the process should write to the disk
    bool _write_to_disk;
    /// A mutex to prevent the rest server callback from overwriting data currently in use
    std::mutex rest_callback_mutex;
    /// String to hold endpoint name
    string endpoint;
};

#endif
