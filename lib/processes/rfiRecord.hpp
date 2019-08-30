/**
 * @file rfiRecorder.hpp
 * @brief Contains RFI data recorder for SK estimates in kotekan.
 *  - rfiRecord : public kotekan::Stage
 */

#ifndef RFI_RECORD_H
#define RFI_RECORD_H

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "powerStreamUtil.hpp"
#include "restServer.hpp"

#include <sys/socket.h>

/*
 * @class rfiRecord
 * @brief Consumer ``kotekan::Stage`` which consumes and record a buffer filled with
 * spectral kurtosis estimates.
 *
 * This stage reads spectral kurtosis estimate from the GPU/CPU and records them to file. The
 * stage will create sub driectories for each stream and data acquisition session. The stage
 * will record a new file every 1024 frames. An info file will be availiable in each directory
 * indicating all the dataset parameters.
 *
 * @par Buffers
 * @buffer rfi_in       The kotekan buffer containing spectral kurtosis estimates to be read by the
 * stage.
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
 * @conf   rfi_combined         Bool (default true). Whether or not the kurtosis measurements
 * include an input sum.
 * @conf   total_links          Int (default 1). Number of FPGA links per buffer.
 * @conf   write_to             String . Path to directory where the stage will record data.
 * @conf   write_to_disk        Bool (default false). Whether or not the proccess will wirte to
 * disk.
 *
 * @author Jacob Taylor
 */
class rfiRecord : public kotekan::Stage {
public:
    // Constructor
    rfiRecord(kotekan::Config& config, const string& unique_name,
              kotekan::bufferContainer& buffer_container);
    // Deconstructor, cleans up / does nothing
    virtual ~rfiRecord();
    // Primary loop, reads buffer and sends out UDP stream
    void main_thread() override;
    // Callback function called by rest server
    void rest_callback(kotekan::connectionInstance& conn, json& json_request);

private:
    /// Kotekan buffer containing kurtosis estimates
    struct Buffer* rfi_buf;
    // General Config Parameters
    /// Number of elements (2048 for CHIME or 256 for Pathfinder)
    uint32_t _num_elements;
    /// Number of frequencies per GPU (1 for CHIME or 8 for Pathfinder)
    uint32_t _num_local_freq;
    /// Total number of frequencies (1024)
    uint32_t _num_freq;
    /// Number of time samples per frame (Usually 32768 or 49152)
    uint32_t _samples_per_data_set;
    /// How many frames to write into the file befor starting a newone.
    uint32_t _frames_per_file;
    // RFI config parameters
    /// The kurtosis step (How many timesteps per kurtosis estimate)
    uint32_t _sk_step;
    /// Flag for element summation in kurtosis estimation process
    bool _rfi_combined;
    // Stage-specific config parameters
    /// The total number of links processed by gpu
    uint32_t _total_links;
    /// The current file index
    uint32_t file_num;
    /// Where to record the RFI data
    string _write_to;
    /// Whether or not the stage should write to the disk
    bool _write_to_disk;
    /// A mutex to prevent the rest server callback from overwriting data currently in use
    std::mutex rest_callback_mutex;
    /// String to hold endpoint name
    string endpoint;
};

#endif
