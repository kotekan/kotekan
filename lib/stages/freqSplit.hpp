/*****************************************
@file
@brief Stage for splitting visibility data by frequency.
- freqSplit : public kotekan::Stage
*****************************************/
#ifndef FREQ_SPLIT_HPP
#define FREQ_SPLIT_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"
#include "dataset.hpp" // for dset_id_t

#include <array>   // for array
#include <future>  // for future
#include <string>  // for string
#include <utility> // for pair
#include <vector>  // for vector


/**
 * @class freqSplit
 * @brief Separate a visBuffer stream into two by selecting frequencies in the upper and lower half
 * of the band.
 *
 * This task takes data coming out of a visBuffer stream and separates it into
 * two streams. It selects which frames to copy to which buffer by assigning
 * frequencies in the upper and lower half of the CHIME band to different buffer
 * streams.
 *
 * @par Buffers
 * @buffer in_buf The buffer to be split
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 * @buffer out_bufs The two buffers containing the respective upper or lower band frequencies
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf split_freq             Double. Frequency to split the incoming buffer
 *                              at. Lower frequencies got to the first output
 *                              buffer, equal and higher frequencies go to the
 *                              second. Default 512.
 *
 * @todo Generalise to arbitary frequency splits.
 * @author Mateus Fandino
 */
class freqSplit : public kotekan::Stage {

public:
    // Default constructor
    freqSplit(kotekan::Config& config, const std::string& unique_name,
              kotekan::bufferContainer& buffer_container);

    // Main loop for the stage
    void main_thread() override;

private:
    /// adds states and datasets and gets new output dataset IDs from manager
    std::array<dset_id_t, 2> change_dataset_state(dset_id_t input_dset_id);

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<Buffer*, unsigned int>> out_bufs;
    Buffer* in_buf;

    std::future<std::array<dset_id_t, 2>> _output_dset_id;

    // config values
    double _split_freq;
};


#endif
