/*****************************************
@file
@brief Generate fake data in gpu buffer format.
- fakeGpuBuffer : public KotekanProcess
*****************************************/
#ifndef FAKE_GPU_BUFFER_HPP
#define FAKE_GPU_BUFFER_HPP

#include "buffer.h"
#include "KotekanProcess.hpp"


/**
 * @class fakeGpuBuffer
 * @brief Generate fake data in gpu buffer format.
 *
 * Produce data formatted with the gpu buffer blocked packing.
 *
 * @par Buffers
 * @buffer out_buf The buffer of fake data.
 *         @buffer_format N2 GPU buffer format
 *         @buffer_metadata chimeMetadata
 *
 * @conf  num_elements      Int. The number of elements (i.e. inputs) in the
 *                          correlator data
 * @conf  block_size        Int. The block size of the packed data
 * @conf  num_eigenvectors  Int. The number of eigenvectors to be stored
 * @conf  freq              Int. The single frequency ID to generate frames for.
 * @conf  cadence           Float. The interval of time (in seconds) between frames.
 *
 * @warning The `stream_id_t` in the metadata is likely to be invalid as it is
 *          generated only such that it is decoded back to the input frequency
 *          id.
 * @author Richard Shaw
 */
class fakeGpuBuffer : public KotekanProcess {
public:
    fakeGpuBuffer(Config& config,
                const string& unique_name,
                bufferContainer &buffer_container);
    ~fakeGpuBuffer();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
private:

    Buffer *output_buf;

    // Parameters read from the config
    int freq;
    float cadence;
    int32_t block_size;
    int32_t num_blocks;
};

#endif
