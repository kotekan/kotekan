/**
 * @file
 * @brife Merge the single frame beam buffer to a buffer with longer timespan.
 *  - mergeBeamBuffer : public kotekan::Stage
 */

#ifndef MERGE_BEAM_BUFFER
#define MERGE_BEAM_BUFFER

#include "BeamMetadata.hpp" // for BeamMetadata
#include "Config.hpp"       // for Config
#include "Stage.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


using std::vector;


/**
 * @class mergeBeamBuffer
 * @brief A stage to merge n single beam frame to one single
 *        beam frame.
 *
 * @par Buffers
 * @buffer in_buf Kotekan buffer of raw packets.
 *     @buffer_format Array of @c chars
 * @buffer out_buf Kotekan buffer of compressed lost samples.
 *     @buffer_format Array of @c uint32_t
 *
 * @conf   samples_per_data_set  Int.    No. of samples.
 * @conf   sub_frames_per_merged_frame  Int.   Number of frames to merge.
 *
 * @author Jing
 *
 *
 */


class mergeBeamBuffer : public kotekan::Stage {
public:
    /// Constructor
    mergeBeamBuffer(kotekan::Config& config_, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~mergeBeamBuffer();
    /// Primary loop to wait for buffers, put package together.
    void main_thread() override;

private:
    /// Merged buffer for the merged
    struct Buffer* in_buf;

    /// Frame merged buffer
    struct Buffer* out_buf;


    /// Config variables
    uint32_t _samples_per_data_set;
    uint32_t _sub_frames_per_merged_frame;
};


#endif // MERGE_BEAM_BUFFER_HPP
