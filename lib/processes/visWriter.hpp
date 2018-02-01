/*****************************************
@file
@brief Processes for handling visibility data.
- visTransform : public KotekanProcess
- visDebug : public KotekanProcess
- visWriter : public KotekanProcess

@todo Move processes that require HDF5 into a separate file.
*****************************************/
#ifndef VIS_WRITER_HPP
#define VIS_WRITER_HPP

#include <unistd.h>
#include "fpga_header_functions.h"
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visFile.hpp"
#include "errors.h"
#include "util.h"
#include "visUtil.hpp"


/**
 * @class visTransform
 * @brief Merge a set of GPU buffers into a single visBuffer stream.
 *
 * This task takes data coming out of a collecton of GPU streams and merges and
 * reformats it into a single stream in the new visBuffer format that is used
 * for the receiver.
 *
 * @par Buffers
 * @buffer in_bufs The set of buffers coming out the GPU buffers
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf The merged and transformed buffer
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf  num_elements      Int. The number of elements (i.e. inputs) in the
 *                          correlator data (read from "/")
 * @conf  block_size        Int. The block size of the packed data (read from "/")
 * @conf  num_eigenvectors  Int. The number of eigenvectors to be stored
 * @conf  input_reorder     Array of [int, int, string]. The reordering mapping.
 *                          Only the first element of each sub-array is used and
 *                          it is the the index of the input to move into this
 *                          new location. The remaining elements of the subarray
 *                          are for correctly labelling the input in
 *                          ``visWriter``.
 *
 * @author Richard Shaw
 */
class visTransform : public KotekanProcess {

public:

    // Default constructor
    visTransform(Config &config,
                const string& unique_name,
                bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    // Main loop for the process
    void main_thread();

private:

    // Parameters saved from the config files
    size_t num_elements, num_eigenvectors, block_size;

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<Buffer*, unsigned int>> input_buffers;
    Buffer * output_buffer;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

};


/**
 * @class visDebug
 * @brief Output some useful properties about the buffer for debugging
 *
 * The output is produced by calling ``visFrameView::summary``
 *
 * @par Buffers
 * @buffer in_buf The buffer to debug
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @author Richard Shaw
 */
class visDebug : public KotekanProcess {

public:
    visDebug(Config &config,
             const string& unique_name,
             bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:

    Buffer * buffer;
};


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
 * The output is written into the CHIME N^2 HDF% format version 3.0.
 *
 * @par Buffers
 * @buffer in_buf The buffer streaming data to write
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf   node_mode        Bool (default: true). Run in ``node_mode`` or not.
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


private:

    /// Setup the acquisition
    void init_acq();

    /// Given a list of stream_ids infer the frequencies in the file, and create
    /// a mapping from id to frequency index
    void setup_freq(const std::vector<uint32_t>& freq_ids);

    // Parameters saved from the config files
    size_t num_freq;
    std::string root_path;
    std::string instrument_name;

    // The current file of visibilities that we are writing
    std::unique_ptr<visFileBundle> file_bundle;

    /// Input buffer to read from
    Buffer * buffer;

    /// The list of frequencies and inputs that gets written into the index maps
    /// of the HDF5 files
    std::vector<freq_ctype> freqs;
    std::vector<input_ctype> inputs;

    /// The mapping from frequency bin id to output frequency index
    std::map<uint32_t, uint32_t> freq_map;

    /// A unique ID for the chunk (i.e. frequency set)
    uint32_t chunk_id;

    /// Params for supporting old node based HDF5 writing scheme
    bool node_mode;
    std::vector<int> freq_id_list;
};

#endif
