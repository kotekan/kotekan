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
#include "visFile.hpp"           // for visFileBundle, visCalFileBundle
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

class HFBWriter : public BaseWriter {
public:
    HFBWriter(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);

    void main_thread() override;

protected:
    /// Setup the acquisition
    // NOTE: must be called from with a region locked by acqs_mutex
    void init_acq(dset_id_t ds_id, std::map<std::string, std::string> metadata) override;

    /// Construct the set of metadata
    std::map<std::string, std::string> make_metadata(dset_id_t ds_id) override;

    /// Gets states from the dataset manager and saves some metadata
    void get_dataset_state(dset_id_t ds_id) override;

    void write_data(const FrameView& frame, kotekan::prometheus::Gauge& write_time_metric,
                    std::unique_lock<std::mutex>& acqs_lock) override;
};

#endif
