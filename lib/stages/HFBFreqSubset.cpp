#include "HFBFreqSubset.hpp"

#include "Config.hpp"          // for Config
#include "HFBFrameView.hpp"    // for HFBFrameView
#include "Hash.hpp"            // for operator<
#include "Stage.hpp"           // for Stage
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t
#include "visUtil.hpp"         // for frameID

#include <cstdint> // for uint32_t
#include <future>  // for future
#include <map>     // for map
#include <utility> // for pair

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(HFBFreqSubset);


HFBFreqSubset::HFBFreqSubset(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    FreqSubset(config, unique_name, buffer_container) {}

void HFBFreqSubset::copy_dataset_id(dset_id_t dataset_id, frameID input_frame_id,
                                    frameID output_frame_id) {

    // Copy frame and create view
    auto output_frame = HFBFrameView::copy_frame(in_buf, input_frame_id, out_buf, output_frame_id);

    // Wait for the dataset ID for the outgoing frame
    if (change_dset_fut.valid())
        change_dset_fut.wait();

    output_frame.dataset_id = dset_id_map.at(dataset_id);
}

std::pair<dset_id_t, uint32_t> HFBFreqSubset::get_frame_data(frameID input_frame_id) {

    auto frame = HFBFrameView(in_buf, input_frame_id);
    return {frame.dataset_id, frame.freq_id};
}
