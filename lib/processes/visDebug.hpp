/*****************************************
@file
@brief Process for debugging visibility buffers.
- visDebug : public kotekan::Stage
*****************************************/
#ifndef VISDEBUG_HPP
#define VISDEBUG_HPP

#include "Config.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "stage.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>


/**
 * @class visDebug
 * @brief Output some useful properties about the buffer for debugging
 *
 * The output is produced by calling ``visFrameView::summary``
 *
 * @par Buffers
 * @buffer in_buf The buffer to debug
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @par Metrics
 * @metric kotekan_visdebug_frame_total
 *         The total frames seen per frequency and dataset (given as labelled).
 *
 * @author Richard Shaw
 */
class visDebug : public kotekan::Stage {

public:
    visDebug(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    void main_thread() override;

private:
    Buffer* in_buf;

    // A (freq_id, dataset_id) pair
    using fd_pair = typename std::pair<uint32_t, uint64_t>;

    // Count the number of frames received for every {freq_id, dataset_id}
    std::map<fd_pair, uint64_t> frame_counts;
};

#endif // VISDEBUG_HPP