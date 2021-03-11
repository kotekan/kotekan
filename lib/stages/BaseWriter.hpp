/*****************************************
@file
@brief Base class for writer stage.
- BaseWriter : public
*****************************************/
#ifndef BASE_WRITER_HPP
#define BASE_WRITER_HPP

#include "Config.hpp"            // for Config
#include "FrameView.hpp"         // for FrameView
#include "Stage.hpp"             // for Stage
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t
#include "prometheusMetrics.hpp" // for Counter, MetricFamily, Gauge
#include "visFile.hpp"           // for visFileBundle
#include "visUtil.hpp"           // for movingAverage, time_ctype

#include <cstdint> // for uint32_t, int64_t
#include <map>     // for map
#include <memory>  // for shared_ptr, unique_ptr
#include <set>     // for set
#include <stdio.h> // for size_t
#include <string>  // for string

/**
 * @class BaseWriter
 * @brief Generic class to write a buffer as raw data.
 *
 * All classes which inherit from this should provide the following API:
 *
 * make_metadata(dset_id_t ds_id);
 * get_dataset_state(dset_id_t ds_id);
 * write_data(const FrameView& frame);
 *
 * This stage writes out the data it receives with minimal processing.
 * Removing certain fields from the output must be done in a prior
 * transformation. See `removeEv` for instance.
 *
 * To obtain the metadata about the stream received usage of the datasetManager
 * is required.
 *
 * This stage will check that git hash of the data source (obtained from the
 * metadataState) matches the current version. Depending on the value of
 * `ignore_version` a mismatch will either generate a warning, or cause the
 * mismatched data to be dropped.
 *
 * The output files will write out the dataset ID of every piece of data
 * written. This allows the dataset ID to change on the incoming stream without
 * requiring a new file or acquisition to be started. However, some types of
 * state change will always cause a new acquisition to be started. By default
 * these are the structural parameters `input`, `frequencies`, `products`, and
 * `stack` along with `gating` and `metadata`. This list can be *added* to
 * using the config variable `critical_states`. Any state change not considered
 * critical will cause an updated ID to be written into the file, but the
 * acquisition will continue as normal.
 *
 * The output is written a specified format. Either the CHIME N^2 HDF5 format
 * version 3.1.0, or the raw format which can be processed into that format by
 * gossec.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf   file_type        String. Type of file to write. One of 'hdf5',
 *                          'hdf5fast' or 'raw'.
 * @conf   root_path        String. Location in filesystem to write to.
 * @conf   instrument_name  String (default: chime). Name of the instrument
 *                          acquiring data (if ``node_mode`` the hostname is
 *                          used instead)
 * @conf   file_length      Int (default 1024). Maximum number of samples to
 *                          write into a file.
 * @conf   window           Int (default 20). Number of samples to keep active
 *                          for writing at any time.
 * @conf   acq_timeout      Double (default 300). Close acquisitions when they
 *                          have been inactive this long (in seconds).
 * @conf   ignore_version   Bool (default False). If true, a git version
 *                          mistmatch will generate a warning, if false, it
 *                          will cause data to be dropped. This should only be
 *                          set to true when testing.
 * @conf   critical_states  List of strings. A list of state types to consider
 *                          critical. That is, if they change in the incoming
 *                          data stream then a new acquisition will be started.
 *
 * @par Metrics
 * @metric kotekan_writer_write_time_seconds
 *         The write time of the raw writer. An exponential moving average over ~10
 *         samples.
 * @metric kotekan_writer_late_frame_total
 *         The number of frames dropped while attempting to write as they are too late.
 * @metric kotekan_writer_bad_dataset_frame_total
 *         The number of frames dropped as they belong to a bad dataset.
 *
 * @author Richard Shaw and James Willis
 **/
class BaseWriter : public kotekan::Stage {
public:
    BaseWriter(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    void main_thread() override;

    /// Why was a frame dropped?
    enum class droppedType {
        late,       // Data arrived too late
        bad_dataset // Dataset ID issues
    };

protected:
    /// Write frame
    void write_frame(const FrameView& frame, dset_id_t dataset_id, uint32_t freq_id,
                     time_ctype time);

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

        // Data size
        int64_t frame_size = -1;

        /// Last update
        double last_update;
    };

    /// The set of open acquisitions, keyed by the dataset_id. Multiple
    /// dataset_ids may point to the same acquisition, and these acquisitions are
    /// shared with `acqs_fingerprint`
    std::map<dset_id_t, std::shared_ptr<acqState>> acqs;

    // Parameters saved from the config files
    std::string instrument_name;

    // Acquisition name format
    std::string acq_fmt;

    // File name format
    std::string file_fmt;

private:
    /// Construct the set of metadata
    virtual std::map<std::string, std::string> make_metadata(dset_id_t ds_id) = 0;

    /// Gets states from the dataset manager and saves some metadata
    virtual void get_dataset_state(dset_id_t ds_id) = 0;

    /// Write data using FrameView
    virtual void write_data(Buffer* in_buf, int frame_id) = 0;

    /// Setup the acquisition
    void init_acq(dset_id_t ds_id);

    /// Close inactive acquisitions
    void close_old_acqs();

    /**
     * Check git version.
     *
     * @param  ds_id  Dataset ID.
     *
     * @return        False if there's a mismatch. Always returns true if
     *                `ignore_version` is set.
     **/
    bool check_git_version(dset_id_t ds_id);

    /// The set of open acquisitions, keyed by fingerprint. These are shared with
    /// `acqs`.
    std::map<fingerprint_t, std::shared_ptr<acqState>> acqs_fingerprint;

    /// Translate droppedTypes to string description for prometheus
    static std::map<droppedType, std::string> dropped_type_map;

    /// List of states that will cause a new acq
    std::set<std::string> critical_state_types;

private:
    // Parameters saved from the config files
    std::string root_path;
    std::string file_type; // Type of the file we are writing
    size_t file_length;
    size_t window;
    bool ignore_version;
    double acq_timeout;

    /// Input buffer to read from
    Buffer* in_buf;

    /// Next sweep
    double next_sweep = 0.0;

    /// Keep track of the average write time
    movingAverage write_time;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& late_frame_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& bad_dataset_frame_counter;
    kotekan::prometheus::Gauge& write_time_metric;
};

#endif
