/**
 * @file
 * @brief Displays CHIME metadata for a given buffer.
 *  - chimeMetadataDump : public KotekanProcess
 */
#ifndef CHIME_METADATA_DUMP_HPP
#define CHIME_METADATA_DUMP_HPP

#include "KotekanProcess.hpp"
#include "buffer.h"
#include "errors.h"
#include "util.h"

#include <unistd.h>

/**
 * @class chimeMetadataDump
 * @brief Displays CHIME metedata for a given buffer
 *
 * This is a simple process which prints (via the @c INFO mechanism)
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

class chimeMetadataDump : public kotekan::KotekanProcess {
public:
    /// Constructor.
    chimeMetadataDump(kotekan::Config& config, const string& unique_name,
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
