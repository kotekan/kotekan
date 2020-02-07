/**
 * @file
 * @brief Compresses the lost samples buffer.
 *  - compressLostSamples : public kotekan::Stage
 */

#ifndef COMPRESS_LOST_SAMPLES_PROCESS
#define COMPRESS_LOST_SAMPLES_PROCESS

#include "Stage.hpp"

#include <vector>

using std::vector;

/**
 * @class compressLostSamples
 * @brief Compresses the lost samples buffer by checking samples in blocks of num_sub_freqs * 3, if
 * any are flagged place a 1 in the compressed lost samples frame. samples_per_data_set /
 * num_sub_freq / 3 = 49152 / 128 / 3
 *
 * @par Buffers
 * @buffer in_buf Kotekan buffer of lost samples.
 *     @buffer_format Array of @c chars
 * @buffer out_buf Kotekan buffer of compressed lost samples.
 *     @buffer_format Array of @c chars
 *
 * @conf   samples_per_data_set Int. No. of samples.
 * @conf   num_sub_freqs  Int. No. of sub frequencies (should be 128).
 *
 * @author James Willis
 *
 */

class compressLostSamples : public kotekan::Stage {
public:
    /// Constructor.
    compressLostSamples(kotekan::Config& config_, const std::string& unique_name,
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
