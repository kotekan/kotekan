/*****************************************
@file
@brief Base class for VisWriter stage.
- BaseWriter : public
*****************************************/
#ifndef BASE_WRITER_HPP
#define BASE_WRITER_HPP

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

class BaseWriter : public kotekan::Stage {
public:
    BaseWriter(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Why was a frame dropped?
    enum class droppedType {
        late,       // Data arrived too late
        bad_dataset // Dataset ID issues
    };

protected:
    /// Setup the acquisition
    // NOTE: must be called from with a region locked by acqs_mutex
    virtual void init_acq(dset_id_t ds_id, std::map<std::string, std::string> metadata) = 0;

    /// Construct the set of metadata
    virtual std::map<std::string, std::string> make_metadata(dset_id_t ds_id) = 0;

    /// Close inactive acquisitions
    virtual void close_old_acqs();

    /// Gets states from the dataset manager and saves some metadata
    virtual void get_dataset_state(dset_id_t ds_id) = 0;

    /**
     * Check git version.
     *
     * @param  ds_id  Dataset ID.
     *
     * @return        False if there's a mismatch. Always returns true if
     *                `ignore_version` is set.
     **/
    bool check_git_version(dset_id_t ds_id);

    virtual void write_data(const FrameView& frame, kotekan::prometheus::Gauge& write_time_metric,
                            std::unique_lock<std::mutex>& acqs_lock) = 0;

    // Parameters saved from the config files
    std::string root_path;
    std::string instrument_name;
    std::string file_type; // Type of the file we are writing
    size_t file_length;
    size_t window;
    size_t rollover;
    bool ignore_version;
    double acq_timeout;

    /// Input buffer to read from
    Buffer* in_buf;

    /// Mutex for updating file_bundle (used in for visCalWriter)
    std::mutex write_mutex;

    /// Manage access to the list of acquisitions (again mostly for visCalWriter)
    std::mutex acqs_mutex;

    /// Hold the internal state of an acquisition (one per dataset ID)
    /// Note that we create an acqState even for invalid datasets that we will
    /// reject all data from
    struct acqState {

        /// Is the acq invalid? Drops data with this dataset ID.
        bool bad_dataset = false;

        /// The current set of files we are writing
        std::unique_ptr<visFileBundle> file_bundle;

        /// Frequency IDs that we are expecting
        std::map<uint32_t, uint32_t> freq_id_map;

        /// Number of products
        size_t num_vis;

        /// Number of beams
        size_t num_beams;

        /// Last update
        double last_update;
    };

    /// The set of open acquisitions, keyed by the dataset_id. Multiple
    /// dataset_ids may point to the same acquisition, and these acquisitions are
    /// shared with `acqs_fingerprint`
    std::map<dset_id_t, std::shared_ptr<acqState>> acqs;

    /// The set of open acquisitions, keyed by fingerprint. These are shared with
    /// `acqs`.
    std::map<fingerprint_t, std::shared_ptr<acqState>> acqs_fingerprint;

    /// Translate droppedTypes to string description for prometheus
    static std::map<droppedType, std::string> dropped_type_map;

    /// List of states that will cause a new acq
    std::set<std::string> critical_state_types;

    /// Next sweep
    double next_sweep = 0.0;

protected:
    /// Number of products to write and freqency map
    std::future<std::pair<size_t, std::map<uint32_t, uint32_t>>> future_metadata;

    /// Keep track of the average write time
    movingAverage write_time;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& late_frame_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& bad_dataset_frame_counter;
};

#endif
