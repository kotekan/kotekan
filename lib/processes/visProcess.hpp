/*****************************************
@file
@brief Processes for handling visibility data.
- visTransform : public KotekanProcess
- visDebug : public KotekanProcess
*****************************************/
#ifndef VIS_PROCESS_HPP
#define VIS_PROCESS_HPP

#include <cstdint>

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visFile.hpp"
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
 *                          correlator data.
 * @conf  block_size        Int. The block size of the packed data.
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
};


/**
 * @class visAccumulate
 * @brief Accumulate the high rate GPU output into integrated visBuffers.
 *
 * This process will accumulate the GPU output and calculate the within sample
 * variance for weights.
 *
 * @par Buffers
 * @buffer in_buf
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf
 *         @buffer_format visBuffer
 *         @buffer_metadata visMetadata
 *
 * @conf  samples_per_data_set  Int. The number of samples each GPU buffer has
 *                              been integrated for.
 * @conf  num_gpu_frames        Int. The number of GPU frames to accumulate over.
 * @conf  integration_time      Float. Requested integration time in seconds.
 *                              This can be used as an alterative to
 *                              `num_gpu_frames` (which it overrides).
 *                              Internally it picks the nearest acceptable value
 *                              of `num_gpu_frames`.
 *
 * @author Richard Shaw
 */
class visAccumulate : public KotekanProcess {
public:
    visAccumulate(Config& config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~visAccumulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;

    uint32_t _samples_per_data_set;
    uint32_t _num_gpu_frames;
};

#endif
