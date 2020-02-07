/**
 * @file rfiRecord.hpp
 * @brief Contains RFI data recorder for SK estimates and feed variance
 *  - rfiRecord : public kotekan::Stage
 */

#ifndef RFI_RECORD_H
#define RFI_RECORD_H

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include "json.hpp" // for json

#include <mutex>    // for mutex
#include <stdint.h> // for uint32_t
#include <string>   // for string

/*
 * @class rfiRecord
 * @brief Consumer ``kotekan::Stage`` which consumes and record a buffer filled with
 * spectral kurtosis estimates or feed variance, and writes it to disk.
 *
 * @par Buffers
 * @buffer rfi_in       The kotekan buffer containing spectral kurtosis estimates to be read by the
 * stage.
 *      @buffer_format  Array of @c floats
 *      @buffer_metadata chimeMetadata
 *
 * @conf   samples_per_data_set Int.    Number of time samples in a data set.
 * @conf   frames_per_file      Int.    Number of frames of data to write into each file
 * @conf   updatable_config     String. Path to updatable config block containing the two fields:
 *                                      "write_to_disk", bool for if we should write data
 *                                      "output_dir", string the location to write data files.
 *
 * @author Jacob Taylor and Andre Renard
 */
class rfiRecord : public kotekan::Stage {
public:
    // Constructor
    rfiRecord(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);
    // Deconstructor, cleans up / does nothing
    virtual ~rfiRecord();
    // Primary loop, reads buffer and sends out UDP stream
    void main_thread() override;
    // Callback function called by rest server
    bool config_callback(nlohmann::json& json);

private:
    /// Kotekan buffer containing kurtosis estimates
    struct Buffer* rfi_buf;
    /// Number of samples in each input frame.
    uint32_t _samples_per_data_set;
    /// How many frames to write into the file befor starting a newone.
    uint32_t _frames_per_file;

    /// Writing state control mutex
    std::mutex rest_callback_mutex;
    /// Where to record the RFI data
    std::string _output_dir;
    /// Whether or not the stage should write to the disk
    bool _write_to_disk;
};

#endif
