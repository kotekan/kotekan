/*****************************************
@file
@brief Visibility writer stage.
- VisWriter : public
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include "BaseWriter.hpp" // for BaseWriter

#include <map>    // for map
#include <string> // for string

/**
 * @class VisWriter
 * @brief Stage to write VisBuffer data to disk using BaseWriter.
 *
 * This stage wraps @ref BaseWriter to write visibilities (.raw or HDF5) and
 * provides the metadata required by the vis file implementations.
 */
class VisWriter : public BaseWriter {
public:
    VisWriter(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);

protected:
    /// Construct metadata attributes to embed in the file.
    std::map<std::string, std::string> make_metadata(dset_id_t ds_id) override;

    /// Populate frequency lookup for the incoming dataset.
    void get_dataset_state(dset_id_t ds_id) override;

    /// Write VisFrameView data into the BaseWriter machinery.
    void write_data(Buffer* in_buf, int frame_id) override;

private:
    std::string notes;
};

#endif
