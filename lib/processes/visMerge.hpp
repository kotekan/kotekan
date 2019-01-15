/*****************************************
@file
@brief Processes for merging a set of buffers into a single visBuffer stream.
- visMerge : public KotekanProcess
*****************************************/
#ifndef VISMERGE_HPP
#define VISMERGE_HPP

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "bufferContainer.hpp"
#include "buffer.h"

#include <string>
#include <utility>
#include <vector>


/**
 * @class visMerge
 * @brief Merge a set of buffers into a single visBuffer stream.
 *
 * In reality this probably works on any buffer format, though it is only
 * tested against visBuffer data.
 *
 * @par Buffers
 * @buffer in_bufs The set of buffers to merge together.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata.
 * @buffer out_buf The merged output stream.
 *         @buffer_format visBuffer.
 *         @buffer_metadata visMetadata
 *
 * @author Richard Shaw
 */
class visMerge : public KotekanProcess {

public:

    // Default constructor
    visMerge(Config &config,
             const std::string& unique_name,
             bufferContainer &buffer_container);

    // Main loop for the process
    void main_thread() override;

private:

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<Buffer*, unsigned int>> in_bufs;
    Buffer * out_buf;
};

#endif // VISMERGE_HPP