/**
 * @file
 * @brief Config-driven insertion of missing frames
 */

#ifndef TEST_DROP_FRAMES_HPP
#define TEST_DROP_FRAMES_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class TestDropFrames
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
 * @author James Willis, Davor Cubranic
 **/

class TestDropFrames : public kotekan::Stage {
public:
    /// Constructor
    TestDropFrames(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~TestDropFrames() = default;
    void main_thread() override;

private:
    /// Initializes internal variables from config, allocates reorder_map, gain, get metadata buffer
    Buffer* in_buf;
    Buffer* out_buf;

    /// Number of time samples, should be a multiple of 3x128 for FRB, standard ops is 49152
    uint32_t _samples_per_data_set;
    // List of missing frames
    const std::vector<uint32_t> _missing_frames;
    /// Percentage of frames to drop
    const double _drop_frame_chance;
};

#endif
