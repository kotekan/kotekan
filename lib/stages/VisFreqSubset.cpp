#include "VisFreqSubset.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for operator<
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for allocate_new_metadata_object, mark_frame_empty, mark_fram...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, state_id_t, datasetManager
#include "datasetState.hpp"    // for freqState
#include "kotekanLogging.hpp"  // for FATAL_ERROR
#include "visBuffer.hpp"       // for VisFrameView
#include "visUtil.hpp"         // for frameID, freq_ctype, modulo

#include <algorithm>    // for find, max
#include <atomic>       // for atomic_bool
#include <cstdint>      // for uint32_t
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <future>       // for future, async
#include <map>          // for map, operator!=, map<>::mapped_type, map<>::iterator
#include <stdexcept>    // for out_of_range
#include <system_error> // for system_error
#include <utility>      // for pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(VisFreqSubset);


VisFreqSubset::VisFreqSubset(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    FreqSubset(config, unique_name, buffer_container) {}

void VisFreqSubset::copy_dataset_id(dset_id_t dataset_id, frameID input_frame_id,
                                    frameID output_frame_id) {

    // Copy frame and create view
    auto output_frame = VisFrameView::copy_frame(in_buf, input_frame_id, out_buf, output_frame_id);

    // Wait for the dataset ID for the outgoing frame
    if (change_dset_fut.valid())
        change_dset_fut.wait();

    output_frame.dataset_id = dset_id_map.at(dataset_id);
}

std::pair<dset_id_t, uint32_t> VisFreqSubset::get_frame_data(frameID input_frame_id) {

    auto frame = VisFrameView(in_buf, input_frame_id);
    return {frame.dataset_id, frame.freq_id};
}
