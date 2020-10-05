/*****************************************
@file
@brief Absorber HFBWriter stage.
- HFBWriter : public
*****************************************/
#ifndef HFB_WRITER_HPP
#define HFB_WRITER_HPP

#include "BaseWriter.hpp"        // for BaseWriter
#include "Config.hpp"            // for Config
#include "HFBFrameView.hpp"      // for HFBFrameView
#include "Stage.hpp"             // for Stage
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "restServer.hpp"        // for connectionInstance
#include "visFile.hpp"           // for visFileBundle
#include "visUtil.hpp"           // for movingAverage

#include <cstdint>   // for uint32_t
#include <errno.h>   // for ENOENT, errno
#include <future>    // for future
#include <map>       // for map
#include <memory>    // for shared_ptr, unique_ptr
#include <set>       // for set
#include <stdexcept> // for runtime_error
#include <stdio.h>   // for size_t, remove
#include <string>    // for string, operator+
#include <unistd.h>  // for access, F_OK
#include <utility>   // for pair

/**
 * @class HFBWriter
 * @brief Stage to write raw absorber data.
 *
 * This class inherits from the BaseWriter base class and writes raw absorber data
 * @author James Willis
 **/
class HFBWriter : public BaseWriter {
public:
    HFBWriter(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);

protected:
    /// Construct the set of metadata
    std::map<std::string, std::string> make_metadata(dset_id_t ds_id) override;

    /// Gets states from the dataset manager and saves some metadata
    void get_dataset_state(dset_id_t ds_id) override;

    /// Write data using HFBFrameView
    void write_data(Buffer* in_buf, int frame_id) override;
};

#endif
