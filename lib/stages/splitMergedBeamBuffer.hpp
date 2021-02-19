
/**
 * @file
 * @Split the merged beam buffer to single framed buffer.
 *  - splitMergedBeamBuffer : public kotekan::Stage
 */

#ifndef SPLITE_MERGED_BEAM_BUFFER
#define SPLITE_MERGED_BEAM_BUFFER

#include "BeamMetadata.hpp" // for BeamMetadata
#include "Config.hpp"       // for Config
#include "Stage.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector


using std::vector;


/**
 * @class splitMergedBeamBuffer
 * @brief This stage splits the merged beam buffer to a single frame beam buffer.
 *
 * @par Buffers
 * @buffer in_buf Kotekan buffer of merged packets.
 *     @buffer_format Array of @c chars
 * @buffer out_buf Kotekan buffer of single frame beam packets.
 *     @buffer_format Array of @c uint32_t
 *
 * @author Jing Santiago Luo
 *
 *
 */


class splitMergedBeamBuffer : public kotekan::Stage {
public:
    /// Constructor
    splitMergedBeamBuffer(kotekan::Config& config_, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);

    /// Destructor
    virtual ~splitMergedBeamBuffer();

    /// Primary loop to wait for buffers, split the merged beam buffer.
    void main_thread() override;

private:
    /// Merged buffer with multiple FreqIDBeamMetadata and voltage blocks per frame
    struct Buffer* in_buf;

    /// Buffer with single FreqIDBeamMetadata and voltage blocks
    struct Buffer* out_buf;
};

#endif // MERGE_RAW_BUFFER_HPP
