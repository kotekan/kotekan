/*****************************************
@file
@brief Visibility VisWriter stage.
- VisWriter : public
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include "BaseWriter.hpp"      // for BaseWriter
#include "Config.hpp"          // for Config
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t

#include <map>    // for map
#include <string> // for string

/**
 * @class VisWriter
 * @brief Stage to write raw visibility data.
 *
 * This class inherits from the BaseWriter base class and writes raw visibility data
 *
 * @par Buffers
 * @buffer in_buf Input visibility buffer.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 *
 * @conf file_type       String. See BaseWriter (hdf5/hdf5fast/raw).
 * @conf root_path       String. Output directory.
 * @conf file_length     Int. Samples per file.
 * @conf window          Int. Sliding window size.
 * @conf instrument_name String. Instrument name.
 * @conf acq_timeout     Double. Inactivity timeout.
 * @conf ignore_version  Bool. Allow git mismatch.
 * @conf critical_states Array<String>. Extra critical dataset states.
 *
 * @par Example
 * @code
 * vis_writer:
 *   kotekan_stage: VisWriter
 *   in_buf: vis_in
 *   file_type: hdf5fast
 *   root_path: /data/vis
 *   file_length: 1024
 *   window: 20
 *   instrument_name: chime
 *   acq_timeout: 300
 *   ignore_version: false
 *   critical_states: []
 * @endcode
 * @author Richard Shaw and James Willis
 **/
class VisWriter : public BaseWriter {
public:
    VisWriter(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);

protected:
    /// Construct the set of metadata
    std::map<std::string, std::string> make_metadata(dset_id_t ds_id) override;

    /// Gets states from the dataset manager and saves some metadata
    void get_dataset_state(dset_id_t ds_id) override;

    /// Write data using VisFrameView
    void write_data(Buffer* in_buf, int frame_id) override;
};

#endif
