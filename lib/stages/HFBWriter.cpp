
#include "HFBWriter.hpp"

#include "Config.hpp"            // for Config
#include "HFBFrameView.hpp"      // for HFBFrameView
#include "Hash.hpp"              // for Hash, operator<
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "SystemInterface.hpp"   // for get_hostname, get_username
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for datasetManager, dset_id_t
#include "datasetState.hpp"      // for metadataState, freqState, beamState
#include "kotekanLogging.hpp"    // for FATAL_ERROR, ERROR
#include "prometheusMetrics.hpp" // for Metrics
#include "restServer.hpp"        // for HTTP_RESPONSE, connectionInstance, restServer
#include "version.h"             // for get_git_commit_hash
#include "visUtil.hpp"           // for ts_to_double, time_ctype

#include <cstdint>      // for uint32_t, uint64_t
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <future>       // for async, future
#include <map>          // for map, map<>::mapped_type
#include <memory>       // for __shared_ptr_access, shared_ptr
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for out_of_range
#include <string>       // for string, to_string
#include <sys/types.h>  // for uint
#include <system_error> // for system_error
#include <utility>      // for pair
#include <vector>       // for vector

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
    BaseWriter(config, unique_name, buffer_container, "hfb"){};

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
    metadata["system_user"] = get_username();
    metadata["collection_server"] = get_hostname();
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
}

void HFBWriter::write_data(Buffer* in_buf, int frame_id) {

    auto frame = HFBFrameView(in_buf, frame_id);

    // Get time of the frame
    auto time = frame.time;
    uint64_t fpga_seq_start = frame.fpga_seq_start;
    time_ctype t = {fpga_seq_start, ts_to_double(time)};

    write_frame(frame, frame.dataset_id, frame.freq_id, t);
}
