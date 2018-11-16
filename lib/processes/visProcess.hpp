/*****************************************
@file
@brief Processes for handling visibility data.
- visTransform : public KotekanProcess
- visDebug : public KotekanProcess
- visAccumulate : public KotekanProcess
- visMerge : public KotekanProcess
*****************************************/
#ifndef VIS_PROCESS_HPP
#define VIS_PROCESS_HPP

#include <cstdint>
#include <fstream>

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"
#include "datasetManager.hpp"

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
 *                          correlator data.
 * @conf  block_size        Int. The block size of the packed data.
 * @conf  num_ev            Int. The number of eigenvectors to be stored
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
    std::vector<std::pair<Buffer*, unsigned int>> in_bufs;
    Buffer * out_buf;

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
 * @par Metrics
 * @metric kotekan_visdebug_frame_total
 *         The total frames seen per frequency and dataset (given as labelled).
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
    Buffer * in_buf;

    // A (freq_id, dataset_id) pair
    using fd_pair = typename std::pair<uint32_t, uint64_t>;

    // Count the number of frames received for every {freq_id, dataset_id}
    std::map<fd_pair, uint64_t> frame_counts;
};


/**
 * @class visMerge
 * @brief Merge a set of buffers into a single visBuffer stream.
 *
 * In reality this probably works on any buffer format, though it is only
 * tested against visBuffer data.
 *
 * @par Buffers
 * @buffer in_bufs The set of buffers to merge together.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata.
 * @buffer out_buf The merged output stream.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata
 *
 * @author Richard Shaw
 */
class visMerge : public KotekanProcess {

public:

    // Default constructor
    visMerge(Config &config,
             const string& unique_name,
             bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    // Main loop for the process
    void main_thread();

private:

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<Buffer*, unsigned int>> in_bufs;
    Buffer * out_buf;
};


/**
 * @class visCheckTestPattern
 * @brief Checks if the visibility data matches a given expected pattern.
 *
 * Errors are calculated as the norm of the difference between the expected and
 * the actual (complex) visibility value.
 * For bad frames, the following data is written to a csv file specified in the
 * config:
 * fpga_count:  FPGA counter for the frame
 * time:        the frames timestamp
 * freq_id:     the frames frequency ID
 * num_bad:     number of values that have an error higher then the threshold
 * avg_err:     average error of bad values
 * min_err:     minimum error of bad values
 * max_err:     maximum error of bad balues
 * expected:    the visibility value that was expected according to the mode
 *
 * Additionally a report is printed in a configured interval.
 *
 * The modes are defined as follows:
 * `test_pattern_simple`: All visibility values are `1 + 0j`.
 * `test_pattern_freq`: The value `frequencies` defines frequency bins, the
 * visibilities in frames for those defined frequencies will have the values
 * defined in `freq_values` (in the same order). The visibilities in all other
 * frames will have the value, set by `default_val`.
 *
 * @par Buffers
 * @buffer in_buf               The buffer to debug
 *         @buffer_format       visBuffer structured
 *         @buffer_metadata     visMetadata
 * @buffer out_buf              All frames found to contain errors
 *         @buffer_format       visBuffer structured
 *         @buffer_metadata     visMetadata
 *
 * @conf  out_file              String. Path to the file to dump all output in.
 * @conf  report_freq           Int. Number of frames to print a summary for.
 * @conf  default_val           CFloat. Default expected visibility value.
 * @conf  freq_values           Array of CFloat. Expected visibility value for
 * each frequency bin (used in mode `test_pattern_freq`).
 * @conf  frequencies           Array of Float. Frequency bins (used in mode
 * `test_pattern_freq`).
 * @conf  num_freq              Float. Total number of frequencies in the frames
 * (used in mode `test_pattern_freq`).
 * @conf  tolerance             Float. Defines what difference to the expected
 * value is an error.
 * @conf mode                   String. One of `test_pattern_simple` and
 * `test_pattern_freq`.
 *
 * @author Rick Nitsche
 */
class visTestPattern : public KotekanProcess {

public:
    visTestPattern(Config &config,
             const string& unique_name,
             bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq);

    void main_thread();

private:
    Buffer * in_buf;
    Buffer * out_buf;

    // Config parameters
    float tolerance;
    size_t report_freq;
    std::string mode;
    cfloat exp_val;
    std::vector<cfloat> exp_val_freq;
    size_t num_freq;

    // file to dump all info in
    std::ofstream outfile;
    std::string outfile_name;
};


/**
 * @brief Register the initial state of the buffers with the datasetManager.
 *
 * This task tags a stream with a properly allocated dataset_id and adds
 * associated datasetStates to the datasetManager. This adds an empty stackState
 * to the dataset (as in not stacked) and therefore doesn't support registering
 * stacked data.
 *
 * @note If there are no other consumers on this buffer it will be able to do a
 *       much faster zero copy transfer of the frame from input to output
 *       buffer.
 *
 * @par Buffers
 * @buffer in_bufs The untagged data.
 *         @buffer_format visBuffer structured.
 *         @buffer_metadata visMetadata
 * @buffer out_buf The tagged data.
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf freq_ids   Vector of UInt32. Frequency IDs on the stream.
 *                  Default 0..1023.
 *
 * @author Richard Shaw
 */
class registerInitialDatasetState : public KotekanProcess {

public:

    // Default constructor
    registerInitialDatasetState(Config &config,
                                const string& unique_name,
                                bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq) override;

    // Main loop for the process
    void main_thread() override;

private:

    std::vector<std::pair<uint32_t, freq_ctype>> _freqs;
    std::vector<input_ctype> _inputs;
    std::vector<prod_ctype> _prods;

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf;
};

#endif
