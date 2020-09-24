
#include "HFBWriter.hpp"

#include "Config.hpp"            // for Config
#include "HFBFrameView.hpp"      // for HFBFrameView
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "SystemInterface.hpp"   // for get_user_name, get_host_name
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "restServer.hpp"        // for connectionInstance
#include "version.h"             // for get_git_commit_hash
#include "visFile.hpp"           // for visFileBundle
#include "visUtil.hpp"           // for movingAverage

#include <cstdint>   // for uint32_t
#include <errno.h>   // for ENOENT, errno
#include <future>    // for future
#include <map>       // for map
#include <memory>    // for shared_ptr, unique_ptr
#include <set>       // for set
#include <stdexcept> // for runtime_error
#include <stdio.h>   // for size_t, remove
#include <string>    // for string, operator+
#include <unistd.h>  // for access, F_OK
#include <utility>   // for pair

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(HFBWriter);

HFBWriter::HFBWriter(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container) :
    BaseWriter(config, unique_name, buffer_container){};

/// Construct the set of metadata
std::map<std::string, std::string> HFBWriter::make_metadata(dset_id_t ds_id) {

    // Get the metadata state from the dM
    auto& dm = datasetManager::instance();
    const metadataState* mstate = dm.dataset_state<metadataState>(ds_id);

    // Set the metadata that we want to save with the file
    std::map<std::string, std::string> metadata;
    metadata["weight_type"] = mstate->get_weight_type();
    metadata["instrument_name"] = mstate->get_instrument_name();
    metadata["notes"] = ""; // TODO: connect up notes
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = get_user_name();
    metadata["collection_server"] = get_host_name();
    metadata["num_beams"] =
        std::to_string(config.get<uint32_t>(unique_name, "num_frb_total_beams"));
    metadata["num_sub_freqs"] = std::to_string(config.get<uint32_t>(unique_name, "factor_upchan"));

    return metadata;
}

/// Gets states from the dataset manager and saves some metadata
void HFBWriter::get_dataset_state(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();

    // Get all states synchronously.
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    auto bstate_fut = std::async(&datasetManager::dataset_state<beamState>, &dm, ds_id);
    auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>, &dm, ds_id);

    const freqState* fstate = fstate_fut.get();
    const beamState* bstate = bstate_fut.get();
    const metadataState* mstate = mstate_fut.get();

    if (fstate == nullptr || bstate == nullptr || mstate == nullptr) {
        ERROR("Set to not use dataset_broker and couldn't find ancestor of dataset {}. Make "
              "sure there is a stage upstream in the config, that the dataset states. Unexpected "
              "nullptr: ",
              ds_id);
        if (!fstate)
            FATAL_ERROR("freqState is a nullptr");
        if (!bstate)
            FATAL_ERROR("beamState is a nullptr");
        if (!mstate)
            FATAL_ERROR("metadataState is a nullptr");
    }

    // Get a reference to the acq state
    auto acq = acqs.at(ds_id);

    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        acq->freq_id_map[f.first] = ind++;

    acq->frame_size = HFBFrameView::calculate_frame_size(config, unique_name);
}

void HFBWriter::write_data(Buffer* in_buf, int frame_id) {

    auto frame = HFBFrameView(in_buf, frame_id);

    // Get time of the frame
    auto time = frame.time;
    uint64_t fpga_seq_num = frame.fpga_seq_num;
    time_ctype t = {fpga_seq_num, ts_to_double(time)};

    write_frame(frame, frame.dataset_id, frame.freq_id, t, frame.data_size());
}
