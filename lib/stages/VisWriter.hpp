/*****************************************
@file
@brief Visibility VisWriter stage.
- VisWriter : public
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include "BaseWriter.hpp"        // for BaseWriter
#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "restServer.hpp"        // for connectionInstance
#include "visBuffer.hpp"         // for VisFrameView
#include "visFile.hpp"           // for visFileBundle
#include "visUtil.hpp"           // for movingAverage

#include <cstdint>   // for uint32_t
#include <errno.h>   // for ENOENT, errno
#include <future>    // for future
#include <map>       // for map
#include <memory>    // for shared_ptr, unique_ptr
#include <mutex>     // for mutex
#include <set>       // for set
#include <stdexcept> // for runtime_error
#include <stdio.h>   // for size_t, remove
#include <string>    // for string, operator+
#include <unistd.h>  // for access, F_OK
#include <utility>   // for pair

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
    void write_data(Buffer* in_buf, int frame_id, kotekan::prometheus::Gauge& write_time_metric,
                    std::unique_lock<std::mutex>& acqs_lock) override;
};

#endif
