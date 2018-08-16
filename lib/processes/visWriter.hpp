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
 * The output is written into the CHIME N^2 HDF% format version 3.1.0.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf   node_mode        Bool (default: true). Run in ``node_mode`` or not.
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

    /// Given a list of stream_ids infer the frequencies in the file, and create
    /// a mapping from id to frequency index
    void setup_freq(const std::vector<uint32_t>& freq_ids);

    // Parameters saved from the config files
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
    std::vector<int> freq_id_list;

    // Number of eigenvectors to write out
    size_t num_ev;

    /// Number of products to write
    size_t num_prod;

    /// Keep track of the average write time
    movingAverage write_time;

    uint32_t dropped_frame_count = 0;
};

/**
 * @class visCalWriter
 * @brief Extension to visWriter for exporting calibration data.
 *
 * TODO: fill in...
 *
 * @author Tristan Pinsonneault-Marotte
 **/
class visCalWriter : public visWriter {
public:

    visCalWriter(Config &config,
            const string& unique_name,
            bufferContainer &buffer_container);

    // TODO: is it necessary to override?
    //void main_thread() override;

    /// REST endpoint to request swapping buffer files
    void rest_callback(connectionInstance& conn);

protected:

    // Override function to make visCalFileBundle and set its file name
    void make_bundle(std::map<std::string, std::string>& metadata) override;

    std::string acq_name, file_name;

};

#endif
