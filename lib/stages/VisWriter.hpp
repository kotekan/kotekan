/*****************************************
@file
@brief Visibility VisWriter stage.
- VisWriter : public
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include "BaseWriter.hpp"      // for BaseWriter
#include "Config.hpp"          // for Config
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t

#include <map>    // for map
#include <string> // for string

/**
 * @class VisWriter
 * @brief Stage to write raw visibility data.
 *
 * This class inherits from the BaseWriter base class and writes raw visibility data
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
