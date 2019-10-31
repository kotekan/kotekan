/**
 * @file
 * @brief Config-driven insertion of missing frames
 */

#ifndef TEST_DATA_GEN_MISSING_FRAMES_HPP
#define TEST_DATA_GEN_MISSING_FRAMES_HPP

#include "Stage.hpp"
#include "buffer.h"

/**
 * @class TestDataGenMissingFrames
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
 *
 * @author James Willis, Davor Cubranic
 **/

class TestDataGenMissingFrames : public kotekan::Stage {
public:
    /// Constructor
    TestDataGenMissingFrames(kotekan::Config& config, const string& unique_name,
                             kotekan::bufferContainer& buffer_container);
    ~TestDataGenMissingFrames() = default;
    void main_thread() override;

private:
    /// Initializes internal variables from config, allocates reorder_map, gain, get metadata buffer
    struct Buffer* in_buf;
    struct Buffer* out_buf;

    /// Number of time samples, should be a multiple of 3x128 for FRB, standard ops is 49152
    uint32_t _samples_per_data_set;
    // List of missing frames
    std::vector<uint32_t> _missing_frames;
};

#endif
