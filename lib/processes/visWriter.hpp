/*****************************************
@file
@brief Processes for writing visibility data.
- visWriter : public KotekanProcess
- visCalWriter : public KotekanProcess
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include <errno.h>
#include <stdio.h>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp"
#include "restServer.hpp"
#include "visFile.hpp"
#include "visUtil.hpp"


/**
 * @class visWriter
 * @brief Write the data out to an HDF5 file .
 *
 * This process gets writes out the data it receives with minimal processing.
 * Removing certain fields from the output must be done in a prior
 * transformation. See `removeEv` for instance.
 *
 * To obtain the metadata about the stream received usage of the datasetManager
 * is required.
 *
 * This process will check that git hash of the data source (obtained from the
 * metadataState) matches the current version. Depending on the value of
 * `ignore_version` a mismatch will either generate a warning, or cause the
 * mismatched data to be dropped.
 *
 * The output is written into the CHIME N^2 HDF5 format version 3.1.0.
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
 *
 * @par Metrics
 * @metric kotekan_viswriter_write_time_seconds
 *         The write time of the HDF5 writer. An exponential moving average over ~10
 *         samples.
 * @metric kotekan_viswriter_dropped_frame_total
 *         The number of frames dropped while attempting to write.
 *
 * @author Richard Shaw
 */
class visWriter : public KotekanProcess {
public:
    visWriter(Config &config,
              const string& unique_name,
              bufferContainer &buffer_container);

    void main_thread() override;

    /// Why was a frame dropped?
    enum class droppedType {
        late,  // Data arrived too late
        bad_dataset  // Dataset ID issues
    };

protected:

    /// Setup the acquisition
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

    /**
     * Report a dropped frame to prometheus.
     *
     * @param  ds_id    Dataset ID of frame.
     * @param  freq_id  Freq ID of frame.
     * @param  reason   Reason frame was dropped.
     **/
    void report_dropped_frame(dset_id_t ds_id, uint32_t freq_id,
                              droppedType reason);

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
    Buffer * in_buf;

    /// Mutex for updating file_bundle (used in for visCalWriter)
    std::mutex write_mutex;

    /// Hold the internal state of an acquisition (one per dataset ID)
    /// Note that we create an acqState even for invalid datasets that we will
    /// reject all data from
    struct acqState {

        /// Is the acq invalid? Drops data with this dataset ID.
        bool bad_dataset = false;

        /// The current set of files we are writing
        std::unique_ptr<visFileBundle> file_bundle;

        /// Dropped frame counts per freq ID
        std::map<std::pair<uint32_t, droppedType>, uint64_t>
            dropped_frame_count;

        /// Frequency IDs that we are expecting
        std::map<uint32_t, uint32_t> freq_id_map;

        /// Number of products
        size_t num_vis;

        /// Last update
        double last_update;
    };

    /// The set of open acquisitions
    std::map<dset_id_t, acqState> acqs;

    /// Translate droppedTypes to string description for prometheus
    static std::map<droppedType, std::string> dropped_type_map;

private:

    /// Number of products to write and freqency map
    std::future<std::pair<size_t, std::map<uint32_t, uint32_t>>>
    future_metadata;

    /// Keep track of the average write time
    movingAverage write_time;
};

/**
 * @class visCalWriter
 * @brief Extension to visWriter for exporting calibration data.
 *
 * This process is based off visWriter, but is meant for generating a
 * fixed-length ring-buffer-like file for storing the last samples of
 * the calibration data stream. To ensure consistent reads while the
 * process is continuously writing to a file, a REST endpoint is provided
 * that causes the process to stop writing and release the file for
 * reading. It will proceed with writing to a second file, until the
 * endpoint is called again and it moves back to the first file, and so on.
 *
 * @warning Since the writer starts from scratch when it switches files,
 *          if requests that are sent more frequently than the length of
 *          a file, the released file will be partially empty.
 *
 * @par REST Endpoints
 * @endpoint /release_live_file/process name>    ``GET`` Stop writing
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

    visCalWriter(Config &config,
            const string& unique_name,
            bufferContainer &buffer_container);

    ~visCalWriter();

    /// REST endpoint to request swapping buffer files
    void rest_callback(connectionInstance& conn);

protected:

    // Override function to make visCalFileBundle and set its file name
    void init_acq(dset_id_t ds_id) override;

    // Disable closing old acqs
    void close_old_acqs() override {};

    visCalFileBundle* file_cal_bundle;

    std::string acq_name, fname_live, fname_frozen;

    std::string endpoint;

};


inline void check_remove(std::string fname) {
    if (remove(fname.c_str()) != 0) {
        if (errno != ENOENT)
            throw std::runtime_error("Could not remove file " + fname);
    }
}

#endif
