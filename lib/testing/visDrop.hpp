/*****************************************
@file
@brief Process that drops frames.
- visDrop : public KotekanProcess
*****************************************/
#ifndef VISDROP_HPP
#define VISDROP_HPP

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"

#include <stdint.h>
#include <string>
#include <vector>

/**
 * @brief Drops frames based on given criteria (for testing) without telling the
 * datasetManager.
 *
 * For now, frames can be droppen only based on their frequency ID.
 * TODO: Add more options.
 *
 * @conf  freq           Vector of Uint32. Frequency IDs of frames that should be
 *                       dropped. By default none.
 **/
class visDrop : public kotekan::KotekanProcess {
public:
    // Default constructor
    visDrop(kotekan::Config& config, const string& unique_name,
            kotekan::bufferContainer& buffer_container);

    // Main loop for the process
    void main_thread() override;

private:
    // config parameters
    std::vector<uint32_t> drop_freqs;

    // Buffers
    Buffer* buf_out;
    Buffer* buf_in;
};

#endif /* VISDROP_HPP */
