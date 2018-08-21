/*****************************************
@file
@brief Processes for writing visibility data.
- visWriter : public KotekanProcess
- visCalWriter : public KotekanProcess
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include <cstdint>

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visFile.hpp"
#include "visUtil.hpp"
#include "restServer.hpp"

/**
 * @class visWriter
 * @brief Write the data out to an HDF5 file .
 *
 * This process operates in two modes, ``node_mode`` where it runs on a per-GPU
 * node basis (inferring its frequency selection from that) and writes a new
 * acquisition per node. Alternatively it can be run more generally, receiving
 * and writing arbitrary frequencies, but it must be given the frequency list in
 * the config.
 *
 * The products we are outputting must be specified correctly. This is done
 * using the same configuration parameters as `prodSubset`. If not explicitly
 * set `all` products is assumed.
 *
 * The output is written into the CHIME N^2 HDF5 format version 3.1.0.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf   node_mode        Bool (default: false). Run in ``node_mode`` or not.
 * @conf   file_type        String. Type of file to write. One of 'hdf5',
 *                          'hdf5fast' or 'raw'.
 * @conf   root_path        String. Location in filesystem to write to.
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
 * @conf   write_ev         Bool (default: false). Write out the eigenvalues/vectors.
 * @conf   num_ev           Int. Only needed if `write_ev` is true.
 * @conf   file_length      Int (default 1024). Maximum number of samples to
 *                          write into a file.
 * @conf   window           Int (default 20). Number of samples to keep active
 *                          for writing at any time.
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

    void apply_config(uint64_t fpga_seq);

    void main_thread();


protected:
    // The current file of visibilities that we are writing
    std::shared_ptr<visFileBundle> file_bundle;

    // Override to use a visFileBundle child class
    virtual void make_bundle(std::map<std::string, std::string>& metadata);

    /// Setup the acquisition
    void init_acq();

    /// Using the first frequency ID found, and any config parameters, determine
    /// which frequencies will end up in the file
    void setup_freq(uint32_t freq_id);

    // Parameters saved from the config files
    bool use_dataset_manager;
    size_t num_freq;
    std::string root_path;
    std::string instrument_name;
    std::string weights_type;

    // Type of the file we are writing
    std::string file_type;

    // File length and number of samples to keep "active"
    size_t file_length;
    size_t window;
    size_t rollover;

    /// Input buffer to read from
    Buffer * in_buf;

    /// The list of frequencies and inputs that gets written into the index maps
    /// of the HDF5 files
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;

    /// The mapping from frequency bin id to output frequency index
    std::map<uint32_t, uint32_t> freq_map;

    /// A unique ID for the chunk (i.e. frequency set)
    uint32_t chunk_id;

    // Vector of products if options to restrict them are present
    std::vector<prod_ctype> prods;

    /// Params for supporting old node based HDF5 writing scheme
    bool node_mode;
    std::vector<uint32_t> freq_id_list;

    // Number of eigenvectors to write out
    size_t num_ev;

    /// Number of products to write
    size_t num_prod;

    /// Keep track of the average write time
    movingAverage write_time;

    uint32_t dropped_frame_count = 0;

    /// Mutex for updating file_bundle (used in for visCalWriter)
    std::mutex write_mutex;
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
 * @endpoint /release_cal_file ``GET`` Stop writing and make a file available
 *           for reading. Responds with a path to the file.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf   root_path        String. Location in filesystem to write to.
 * @conf   file_name        String. Name of file to buffer data in (omit ext).
 * @conf   frozen_file_name String. Name of file to release for reading (omit ext).
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
 * @conf   write_ev         Bool (default: false). Write out the eigenvalues/vectors.
 * @conf   num_ev           Int. Only needed if `write_ev` is true.
 * @conf   file_length      Int (default 1024). Fixed length of the ring file
 *                          in number of time samples.
 * @conf   window           Int (default 20). Number of samples to keep active
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
    void make_bundle(std::map<std::string, std::string>& metadata) override;

    std::shared_ptr<visCalFileBundle> file_cal_bundle;

    std::string acq_name, file_name, frozen_file_name;

    std::string endpoint;

};

inline void check_remove(std::string fname) {
    if (remove(fname.c_str()) != 0) {
        throw std::runtime_error("Could not remove file " + fname);
    }
}

inline void check_rename(std::string src, std::string dest) {
    if (rename(src.c_str(), dest.c_str()) != 0) {
        throw std::runtime_error("Could not move file " + src
                + " to " + dest);
    }
}

#endif
