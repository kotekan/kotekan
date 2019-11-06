/*****************************************
@file
@brief Stage for debugging visibility buffers.
- visDebug : public kotekan::Stage
*****************************************/
#ifndef VISDEBUG_HPP
#define VISDEBUG_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"

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
 * @conf    output_period    Int. Print output every `output_period` frames.
 *                           Default is every 1000 frames.
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

    // Output every set number of frames
    int _output_period;
};

#endif // VISDEBUG_HPP
