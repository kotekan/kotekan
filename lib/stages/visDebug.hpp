/*****************************************
@file
@brief Stage for debugging visibility buffers.
- visDebug : public kotekan::Stage
*****************************************/
#ifndef VISDEBUG_HPP
#define VISDEBUG_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <string> // for string


/**
 * @class visDebug
 * @brief Output some useful properties about the buffer for debugging
 *
 * The output is produced by calling ``VisFrameView::summary``
 *
 * @par Buffers
 * @buffer in_buf The buffer to debug
 *         @buffer_format VisBuffer structured
 *         @buffer_metadata VisMetadata
 *
 * @conf    output_period    Int. Print output every `output_period` frames.
 *                           Default is every 1000 frames.
 *
 * @par Metrics
 * @metric kotekan_visdebug_frames_by_freq_total
 *         The total frames seen per frequency.
 * @metric kotekan_visdebug_frames_by_dataset_total
 *         The total frames seen per dataset.
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
