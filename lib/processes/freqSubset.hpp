/*****************************************
@file
@brief Process for subsetting visibility data by frequency.
- freqSubset : public KotekanProcess
*****************************************/
#ifndef FREQ_SUBSET_HPP
#define FREQ_SUBSET_HPP

#include <stdint.h>
#include <future>
#include <string>
#include <vector>

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"
#include "datasetManager.hpp"


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
 * @conf  subset_list           Vector of Int. The list of frequencies that go
 *                              in the subset.
 * @conf  ds_manage_timeout_ms  Int. Time (in ms) before dropping the current
 *                              input frame when waiting for the datasetManager.
 *                              Default 10000.
 *
 * @metric kotekan_dataset_manager_dropped_frame_count
 *        The number of frames dropped while waiting for the dataset manager.
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
    void main_thread() override;

private:
    /// adds state and dataset and gets a new output dataset ID from manager
    dset_id_t change_dataset_state(dset_id_t input_dset_id,
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
    uint64_t _ds_manage_timeout_ms;
};


#endif