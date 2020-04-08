/*****************************************
@file
@brief Stage that drops frames.
- visDrop : public Stage
*****************************************/
#ifndef VISDROP_HPP
#define VISDROP_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @brief Drops frames based on given criteria (for testing) without telling the
 * datasetManager.
 *
 * For now, frames can be droppen only based on their frequency ID.
 * TODO: Add more options.
 *
 * @conf  freq           Vector of Uint32. Frequency IDs of frames that should be
 *                       dropped. By default none.
 * @conf  frac_lost      Float. If > 0, instead of dropping the frame, subtract
 *                       this fraction of FPGA samples from total.
 * @conf  frac_rfi       Float. Set `VisFrameView.rfi_total` to this value.
 *                       Must be <= `frac_lost`.
 **/
class visDrop : public kotekan::Stage {
public:
    // Default constructor
    visDrop(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_container);

    // Main loop for the stage
    void main_thread() override;

private:
    // config parameters
    std::vector<uint32_t> drop_freqs;
    float frac_rfi;
    float frac_lost;

    // Buffers
    Buffer* buf_out;
    Buffer* buf_in;
};

#endif /* VISDROP_HPP */
