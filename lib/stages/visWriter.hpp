/*****************************************
@file
@brief Stages for writing visibility data.
- visWriter : public kotekan::Stage
- visCalWriter : public kotekan::Stage
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include "Config.hpp"            // for Config
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


/**
 * @class visWriter
 * @brief Write the data out to an HDF5 file .
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
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
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
 * @metric kotekan_viswriter_write_time_seconds
 *         The write time of the HDF5 writer. An exponential moving average over ~10
 *         samples.
 * @metric kotekan_viswriter_late_frame_total
 *         The number of frames dropped while attempting to write as they are too late.
 * @metric kotekan_viswriter_bad_dataset_frame_total
 *         The number of frames dropped as they belong to a bad dataset.
 *
 * @author Richard Shaw
 */
class visWriter : public kotekan::Stage {
public:
    visWriter(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);

    void main_thread() override;

    /// Why was a frame dropped?
    enum class droppedType {
        late,       // Data arrived too late
        bad_dataset // Dataset ID issues
    };

protected:
    /// Setup the acquisition
    // NOTE: must be called from with a region locked by acqs_mutex
    virtual void init_acq(dset_id_t ds_id);

    /// Construct the set of metadata
    std::map<std::string, std::string> make_metadata(dset_id_t ds_id);

    /// Close inactive acquisitions
    virtual void close_old_acqs();

    /// Gets states from the dataset manager and saves some metadata
    void get_dataset_state(dset_id_t ds_id);

    /**
     * Check git version.
     *
     * @param  ds_id  Dataset ID.
     *
     * @return        False if there's a mismatch. Always returns true if
     *                `ignore_version` is set.
     **/
    bool check_git_version(dset_id_t ds_id);

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

private:
    /// Number of products to write and freqency map
    std::future<std::pair<size_t, std::map<uint32_t, uint32_t>>> future_metadata;

    /// Keep track of the average write time
    movingAverage write_time;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& late_frame_counter;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Counter>& bad_dataset_frame_counter;
};

/**
 * @class visCalWriter
 * @brief Extension to visWriter for exporting calibration data.
 *
 * This stage is based off visWriter, but is meant for generating a
 * fixed-length ring-buffer-like file for storing the last samples of
 * the calibration data stream. To ensure consistent reads while the
 * stage is continuously writing to a file, a REST endpoint is provided
 * that causes the stage to stop writing and release the file for
 * reading. It will proceed with writing to a second file, until the
 * endpoint is called again and it moves back to the first file, and so on.
 *
 * @warning Since the writer starts from scratch when it switches files,
 *          if requests that are sent more frequently than the length of
 *          a file, the released file will be partially empty.
 *
 * @par REST Endpoints
 * @endpoint /release_live_file/stage name>    ``GET`` Stop writing
 *           and make a file available for reading. Responds with a path to
 *           the file.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf   root_path        String. Location in filesystem to write to.
 * @conf   file_base        String. Base filename to buffer data in (omit ext).
 * @conf   dir_name         String. Name of directory to hold the above files.
 * @conf   node_mode        Bool (default: false). Run in ``node_mode`` or not.
 * @conf   instrument_name  String (default: chime). Name of the instrument
 *                          acquiring data (if ``node_mode`` the hostname is
 *                          used instead)
 * @conf   freq_ids         Array of ints. The ids of the frequencies to write
 *                          out (only needed when not in @c node_mode).
 * @conf   input_reorder    Array of [int, int, string]. A description of the
 *                          inputs. Only the last two elements of each sub-array
 *                          are used and are expected to be @c channel_id and
 *                          @c channel_serial (the first contains the @c adc_id
 *                          used for reordering om ``visTransform``)
 * @conf   weights_type     Indicate what the visibility weights represent, e.g,
 *                          'inverse_var'. Will saved as an attribute in the saved
 *                          file. (default 'unknown')
 * @conf   file_length      Int (default 1024). Fixed length of the ring file
 *                          in number of time samples.
 * @conf   window           Int (default 10). Number of samples to keep active
 *                          for writing at any time.
 *
 * @par Metrics
 * @metric kotekan_viswriter_write_time_seconds
 *         The write time of the HDF5 writer. An exponential moving average over ~10
 *         samples.
 * @metric kotekan_viswriter_dropped_frame_total
 *         The number of frames dropped while attempting to write.
 *
 * @author Tristan Pinsonneault-Marotte
 **/
class visCalWriter : public visWriter {
public:
    visCalWriter(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);

    ~visCalWriter();

    /// REST endpoint to request swapping buffer files
    void rest_callback(kotekan::connectionInstance& conn);

protected:
    // Override function to make visCalFileBundle and set its file name
    void init_acq(dset_id_t ds_id) override;

    // Disable closing old acqs
    void close_old_acqs() override{};

    visCalFileBundle* file_cal_bundle;

    std::string acq_name, fname_live, fname_frozen;

    std::string endpoint;
};


inline void check_remove(std::string fname) {
    // Check if we need to remove anything
    if (access(fname.c_str(), F_OK) != 0)
        return;
    // Remove
    if (remove(fname.c_str()) != 0) {
        if (errno != ENOENT)
            throw std::runtime_error("Could not remove file " + fname);
    }
}

#endif
