/**
 * @file
 * @brief Displays CHIME metadata for a given buffer.
 *  - chimeMetadataDump : public Stage
 */
#ifndef CHIME_METADATA_DUMP_HPP
#define CHIME_METADATA_DUMP_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string


/**
 * @class chimeMetadataDump
 * @brief Displays CHIME metadata for a given buffer
 *
 * This is a simple stage which prints (via the @c INFO mechanism)
 * CHIME metadata from a target buffer.
 *
 * @par Buffers
 * @buffer in_buf Input kotekan buffer to display metadata info for.
 *     @buffer_format Any
 *     @buffer_metadata chimeMetadata
 *
 * @author Andre Renard
 *
 */

class chimeMetadataDump : public kotekan::Stage {
public:
    /// Constructor.
    chimeMetadataDump(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);

    /// Destructor.
    ~chimeMetadataDump();

    /// Primary loop, which waits on input frames, prints the metadata.
    void main_thread() override;

private:
    /// Input kotekanBuffer.
    struct Buffer* in_buf;
};

#endif
