/*****************************************
@file
@brief Processes for splitting and subsetting visibility data by frequency.
- freqSplit : public KotekanProcess
- freqSubset : public KotekanProcess
*****************************************/
#ifndef FREQ_SLICER_HPP
#define FREQ_SLICER_HPP

#include <unistd.h>
#include <future>
#include "buffer.h"
#include "KotekanProcess.hpp"
#include "datasetManager.hpp"


/**
 * @class freqSplit
 * @brief Separate a visBuffer stream into two by selecting frequencies in the upper and lower half of the band.
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
 * @metric kotekan_dataset_manager_dropped_frame_count
 *         The number of frames dropped while attempting to write.
 *
 * @todo Generalise to arbitary frequency splits.
 * @author Mateus Fandino
 */
class freqSplit : public KotekanProcess {

public:

    // Default constructor
    freqSplit(Config &config,
              const string& unique_name,
              bufferContainer &buffer_container);

    // Main loop for the process
    void main_thread();

private:
    /// adds states and datasets and gets new output dataset IDs from manager
    static std::array<dset_id_t, 2>
    change_dataset_state(dset_id_t input_dset_id);

    // Vector of the buffers we are using and their current frame ids.
    std::vector<std::pair<Buffer*, unsigned int>> out_bufs;
    Buffer * in_buf;

    std::future<std::array<dset_id_t, 2>> _output_dset_id;

    // config values
    bool _use_dataset_manager;
};



/**
 * @class freqSubset
 * @brief Outputs a visBuffer stream with a subset of the input frequencies.
 *
 * This task takes data coming out of a visBuffer stream and selects a subset of
 * frequencies to be passed on to the output buffer.
 *
 * @par Buffers
 * @buffer in_buf The original buffer with all frequencies
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 * @buffer out_buf The buffer containing the subset of frequencies
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf  subset_list       Vector of Int. The list of frequencies that go
 *                          in the subset.
 *
 * @metric kotekan_dataset_manager_dropped_frame_count
 *         The number of frames dropped while attempting to write.
 *
 * @author Mateus Fandino
 */
class freqSubset : public KotekanProcess {

public:

    /// Default constructor
    freqSubset(Config &config,
               const string& unique_name,
               bufferContainer &buffer_container);

    /// Main loop for the process
    void main_thread();

private:
    /// adds state and dataset and gets a new output dataset ID from manager
    static dset_id_t change_dataset_state(dset_id_t input_dset_id,
                                          std::vector<uint32_t>& subset_list);

    // List of frequencies for the subset
    std::vector<uint32_t> _subset_list;

    /// Output buffer with subset of frequencies
    Buffer * out_buf;
    /// Input buffer with all frequencies
    Buffer * in_buf;

    // dataset IDs
    std::future<dset_id_t> _output_dset_id;

    // config values
    bool _use_dataset_manager;
};


#endif
