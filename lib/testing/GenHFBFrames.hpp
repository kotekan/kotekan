/**
 * @file
 * @brief Config-driven generation of HFB frames
 */

#ifndef GEN_HFB_FRAMES_HPP
#define GEN_HFB_FRAMES_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <random>   // for default_random_engine, normal_distribution, uniform_int_d...
#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @class GenHFBFrames
 * @brief Config-driven dropping of frames
 *
 * This stage can be interposed between two buffers to drop frames specified in the configuration.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer, to be (selectively) copied over to out_buf
 *     @buffer_format Array of @c any
 *     @buffer_metadata none, or a class derived from @c kotekanMetadata
 * @buffer out_buf Output kotekan buffer, with config-specified frames from in_buf missing
 *     @buffer_format Array of @c datatype
 *     @buffer_metadata none, or a class derived from @c kotekanMetadata
 *
 * @conf   missing_frames @c Vector of UInt32 (Default: empty). Frames to drop.
 * @conf   drop_frame_chance @c Double (Default: 0). Chance of dropping a frame if not in the @c
 *missing_frames list.
 *
 * @author James Willis
 **/

class GenHFBFrames : public kotekan::Stage {
public:
    /// Constructor
    GenHFBFrames(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~GenHFBFrames() = default;
    void main_thread() override;

private:
    void gen_data(float* data, uint32_t* cls_data, std::default_random_engine* gen,
                  std::normal_distribution<float>* gaussian,
                  std::uniform_int_distribution<uint32_t>* rng);

    /// Initializes internal variables from config, allocates metadata
    Buffer* out_buf;
    Buffer* cls_out_buf;

    /// Number of time samples, should be a multiple of 3x128 for FRB, standard ops is 49152
    uint32_t _samples_per_data_set;
    /// Number of samples per HFB frame
    uint32_t _num_samples;
    /// Total number of lost samples per HFB frame
    uint32_t _total_lost_samples;
    /// Index to use when setting initial FPGA seq number
    uint32_t _first_frame_index;
    /// RNG mean
    float _rng_mean;
    /// RNG standard deviation
    float _rng_stddev;
    /// Test pattern
    std::string _pattern;
};

#endif
