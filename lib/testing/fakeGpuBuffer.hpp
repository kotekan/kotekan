/*****************************************
@file
@brief Generate fake data in gpu buffer format.
- fakeGpuBuffer : public KotekanProcess
*****************************************/
#ifndef FAKE_GPU_BUFFER_HPP
#define FAKE_GPU_BUFFER_HPP

#include "buffer.h"
#include "chimeMetadata.h"
#include "KotekanProcess.hpp"
#include "pulsarTiming.hpp"


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
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in
 *                              the correlator data
 * @conf  block_size            Int. The block size of the packed data
 * @conf  num_ev                Int. The number of eigenvectors to be stored
 * @conf  freq_id               Int. The single frequency ID to generate frames
 *                              for.
 * @conf  cadence               Float. The interval of time (in seconds)
 *                              between frames.
 * @conf  pre_accumulate        Bool. Simulate the GPU data before any
 *                              accumulation. This ignores any cadence setting.
 * @conf  samples_per_data_set  Int. FPGA seq ticks per frame. Only use for
 *                              `pre_accumulate`.
 * @conf  mode                  String. Name of the pattern to fill with.
 *                              Descriptions are with the `fill_mode_X`
 *                              methods below.
 * @conf  wait                  Bool. Sleep to try and output data at roughly
 *                              the correct cadence.
 * @conf  num_frames            Exit after num_frames have been produced. If
 *                              less than zero, no limit is applied. Default
 *                              is `-1`.
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
    void main_thread() override;

    /**
     * @brief Fill with a pattern useful for debugging the packing.
     *
     * Fill each element with its block row (real value) and block column
     * (imaginary).
     *
     * @param data      The output frame data to fill.
     * @param frame_num Number of the frame to fill.
     * @param metadata  Metadata of the frame.
     */
    void fill_mode_block(int32_t* data, int frame_num,
                         chimeMetadata* metadata);

    /**
     * @brief Fill with a pattern for testing lost packet renormalisation.
     *
     * Fill each element with its block row (real value) and block column
     * (imaginary). Each frame has a number of lost packets equivalent to the
     * frame number.
     *
     * @param data      The output frame data to fill.
     * @param frame_num Number of the frame to fill.
     * @param metadata  Metadata of the frame.
     */
    void fill_mode_lostsamples(int32_t* data, int frame_num,
                               chimeMetadata* metadata);

    /**
     * @brief Fill with a pattern for debugging the accumulation.
     *
     * Fill each element with its full correlation index (real = row; column =
     * imag), with real and imaginary parts being shifted every 4th frame.
     * 
     * Overall this mode should average to (row + column * J) and the
     * inverse variance should be num_gpu_frames / 8.
     *
     * @param data      The output frame data to fill.
     * @param frame_num Number of the frame to fill.
     * @param metadata  Metadata of the frame.
     */
    void fill_mode_accumulate(int32_t* data, int frame_num,
                              chimeMetadata* metadata);

    /**
     * @brief Fill with a pattern with Gaussian noise with radiometer variance.
     *
     * The underlying inputs are uncorrelated with variance of 1.
     * 
     * @param data      The output frame data to fill.
     * @param frame_num Number of the frame to fill.
     * @param metadata  Metadata of the frame.
     */
    void fill_mode_gaussian(int32_t* data, int frame_num,
                            chimeMetadata* metadata);

    void fill_mode_pulsar(int32_t* data, int frame_num,
                          chimeMetadata* metadata);
private:

    Buffer* out_buf;

    // Parameters read from the config
    int freq;
    float cadence;
    int32_t block_size;
    int32_t num_elements;
    int32_t samples_per_data_set;
    bool pre_accumulate;
    bool wait;
    int32_t num_frames;
    std::vector<float> coeff;
    float dm;
    float pulse_width;  // in s
    float rot_freq; // in Hz
    Polyco * polyco;

    // Function pointer for fill modes
    typedef void(fakeGpuBuffer::*fill_func)(int32_t *, int, chimeMetadata *);

    // A map to look up the modes by name at run time
    std::map<std::string, fill_func> fill_map;

    // The fill function to actually use
    fill_func fill;

};

#endif
