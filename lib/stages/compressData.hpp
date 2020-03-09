/**
 * @file
 * @brief Compress data using a Huffman encoding scheme with N=5.
 *  - compressData : public kotekan::Stage
 */

#ifndef COMPRESS_DATA_PROCESS
#define COMPRESS_DATA_PROCESS

#include "Config.hpp" // for Config
#include "Stage.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h>    // for uint32_t, uint8_t
#include <string>      // for string
#include <sys/types.h> // for ssize_t

/**
 * @class compressData
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
 * @conf   factor_upchan  Int. Upchannelise factor (should be 128).
 *
 * @author James Willis & Alex Roman
 *
 */

class compressData : public kotekan::Stage {
public:
    /// Constructor.
    compressData(kotekan::Config& config_, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~compressData();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;

private:
    int comparef(const float a, const float b);
    void quantize_naive5(const float* in, char* out, const ssize_t n);
    ssize_t huff_size_bound(const ssize_t samples_in);
    ssize_t huff_encode(const char* in, uint32_t* out, const ssize_t n_in);
    ssize_t huff_decompress_bound(const ssize_t size_in);
    ssize_t huff_decode(const uint32_t* in, uint8_t* out, const ssize_t n_in);

    struct Buffer* in_buf;
    struct Buffer* out_buf;

    /// Config variables
    uint32_t _num_frames_to_integrate;
    uint32_t _num_frb_total_beams;
    uint32_t _factor_upchan;

    // the edges of the N=5 bin encoding scheme (determined in quantization.py)
    // -Inf and +Inf are implicit edges for the first and last bins
    const float edges5[4] = {-1.24435754, -0.38228386, 0.38228386, 1.24435754};

    const uint8_t codes[5] = {7, 2, 0, 1, 3}; // value of each huffman code
    const uint32_t codes32[5] = {7, 2, 0, 1,
                                 3};         // value of each code (32 bit dtype for internal use)
    const ssize_t lens[5] = {3, 2, 2, 2, 3}; // bit length of each code

    // hard-coded, used to estimate encoding efficiency
    // we can compute the entropy of an encoding scheme (as a function of N) in quanztiaztion.py
    const float entropy5 = 2.202916387949746;
    const float bitsize = 2.321928094887362;
};

#endif
