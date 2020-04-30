/**
 * @file
 * @brief Compresses the lost samples buffer.
 *  - compressLostSamples : public kotekan::Stage
 */

#ifndef COMPRESS_LOST_SAMPLES_PROCESS
#define COMPRESS_LOST_SAMPLES_PROCESS

#include "Config.hpp" // for Config
#include "Stage.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


using std::vector;

/**
 * @class compressLostSamples
 * @brief Compresses the lost samples buffer into an array of
 *        @c samples_per_data_set/compression_factor values with the number of individual
 *        samples lost in each block of @c compression_factor values.
 *
 * @par Buffers
 * @buffer in_buf Kotekan buffer of lost samples.
 *     @buffer_format Array of @c chars
 * @buffer out_buf Kotekan buffer of compressed lost samples.
 *     @buffer_format Array of @c uint32_t
 *
 * @conf   samples_per_data_set  Int.    No. of samples.
 * @conf   compression_factor    Int.    Number of samples to group.
 * @conf   zero_all_in_group     bool.   If this set to true then one or more samples is lost in
 *                                       a @c compression_factor group, then consider that to be
 *                                       a total of @c compression_factor lost samples for the
 *                                       metadata lost_samples value.
 *
 * @author James Willis, Andre Renard
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
    uint32_t _compression_factor;
    bool _zero_all_in_group;
};

#endif
