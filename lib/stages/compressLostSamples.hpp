/**
 * @file
 * @brief Integrate 21-cm data over frames.
 *  - compressLostSamples : public kotekan::Stage
 */

#ifndef COMPRESS_LOST_SAMPLES_PROCESS
#define COMPRESS_LOST_SAMPLES_PROCESS

#include "Stage.hpp"

#include <vector>

using std::vector;

/**
 * @class compressLostSamples
 * @brief Post-processing engine for output of the CHIME/HFB kernel,
 *        integrates data over 80 frames to create 10s worth of data.
 *        num_beams * num_sub_freq = 1024 * 128
 *
 * This engine sums CHIME/HFB data from 1 GPU stream in each CHIME node,
 * which are stored in the output buffer.
 *
 * @par Buffers
 * @buffer hfb_input_buffer Kotekan buffer feeding data from any GPU.
 *     @buffer_format Array of @c floats
 * @buffer hfb_out_buf Kotekan buffer that will be populated with integrated data.
 *     @buffer_format Array of @c floats
 *
 * @conf   num_frames_to_integrate Int. No. of frames to integrate over.
 * @conf   num_frb_total_beams  Int. No. of total FRB beams (should be 1024).
 * @conf   num_sub_freqs  Int. No. of sub frequencies (should be 128).
 *
 * @author James Willis
 *
 */

class compressLostSamples : public kotekan::Stage {
public:
    /// Constructor.
    compressLostSamples(kotekan::Config& config_, const string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~compressLostSamples();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;

private:
    /// Lost samples buffer
    struct Buffer* in_buf;

    /// Compressed lost samples buffer
    struct Buffer* out_buf;

    /// Config variables
    uint32_t _samples_per_data_set;
    uint32_t _num_sub_freqs;
};

#endif
