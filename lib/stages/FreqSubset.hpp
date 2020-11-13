/*****************************************
@file
@brief Stage for subsetting data by frequency.
- FreqSubset : public kotekan::Stage
*****************************************/
#ifndef FREQ_SUBSET_HPP
#define FREQ_SUBSET_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.h"            // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, state_id_t, fingerprint_t
#include "visUtil.hpp"         // for frameID

#include <future>   // for future
#include <map>      // for map
#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

/**
 * @class FreqSubset
 * @brief Base class to output a buffer stream with a subset of the input frequencies.
 *
 * All classes which inherit from this should provide the following API:
 *
 * copy_dataset_id(dset_id_t dataset_id, frameID input_frame_id, frameID output_frame_id);
 # get_frame_data(frameID input_frame_id);
 *
 * This task takes data coming out of a buffer stream and selects a subset of
 * frequencies to be passed on to the output buffer.
 *
 * @par Buffers
 * @buffer in_buf The original buffer with all frequencies
 *         @buffer_format VisBuffer/HFBBuffer structured
 *         @buffer_metadata VisMetadata/HFBMetadata
 * @buffer out_buf The buffer containing the subset of frequencies
 *         @buffer_format VisBuffer/HFBBuffer structured
 *         @buffer_metadata VisMetadata/HFBMetadata
 *
 * @conf  subset_list           Vector of Int. The list of frequencies that go
 *                              in the subset.
 *
 * @author Mateus Fandino
 */
class FreqSubset : public kotekan::Stage {

public:
    /// Default constructor
    FreqSubset(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffer_container);

    /// Main loop for the stage
    void main_thread() override;

protected:
    /// Output buffer with subset of frequencies
    Buffer* out_buf;
    /// Input buffer with all frequencies
    Buffer* in_buf;

    std::future<void> change_dset_fut;

    // Map for determining the dataset ID to use
    std::map<dset_id_t, dset_id_t> dset_id_map;

private:
    // Copy dataset ID from input to output frame
    virtual void copy_dataset_id(dset_id_t dataset_id, frameID input_frame_id,
                                 frameID output_frame_id) = 0;

    // Get dataset ID and frequency ID from frame
    virtual std::pair<dset_id_t, uint32_t> get_frame_data(frameID input_frame_id) = 0;

    /// adds state and dataset and gets a new output dataset ID from manager
    void change_dataset_state(dset_id_t input_dset_id);

    // List of frequencies for the subset
    std::vector<uint32_t> _subset_list;

    // Map for determining the dataset ID to use
    std::map<fingerprint_t, state_id_t> states_map;
};
#endif
