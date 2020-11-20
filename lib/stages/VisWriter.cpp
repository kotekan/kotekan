
#include "VisWriter.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator<
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "SystemInterface.hpp"   // for get_hostname, get_username
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for datasetManager, dset_id_t
#include "datasetState.hpp"      // for metadataState, freqState, prodState
#include "kotekanLogging.hpp"    // for FATAL_ERROR, ERROR
#include "prometheusMetrics.hpp" // for Metrics
#include "restServer.hpp"        // for HTTP_RESPONSE, connectionInstance, restServer
#include "version.h"             // for get_git_commit_hash
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for ts_to_double, time_ctype

#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <future>       // for async, future
#include <map>          // for map, map<>::mapped_type
#include <memory>       // for __shared_ptr_access, shared_ptr
#include <stdexcept>    // for out_of_range
#include <string>       // for string
#include <sys/types.h>  // for uint
#include <system_error> // for system_error
#include <tuple>        // for get
#include <utility>      // for pair
#include <vector>       // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(VisWriter);

VisWriter::VisWriter(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container) :
    BaseWriter(config, unique_name, buffer_container){};

/// Construct the set of metadata
std::map<std::string, std::string> VisWriter::make_metadata(dset_id_t ds_id) {

    // Get the metadata state from the dM
    auto& dm = datasetManager::instance();
    const metadataState* mstate = dm.dataset_state<metadataState>(ds_id);

    // Set the metadata that we want to save with the file
    std::map<std::string, std::string> metadata;
    metadata["weight_type"] = mstate->get_weight_type();
    metadata["archive_version"] = "NT_3.1.0";
    metadata["instrument_name"] = instrument_name;
    metadata["notes"] = ""; // TODO: connect up notes
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = get_username();
    metadata["collection_server"] = get_hostname();

    return metadata;
}

/// Gets states from the dataset manager and saves some metadata
void VisWriter::get_dataset_state(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();

    // Get all states synchronously.
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, ds_id);
    auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>, &dm, ds_id);

    const metadataState* mstate = mstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const prodState* pstate = pstate_fut.get();

    if (pstate == nullptr || mstate == nullptr || fstate == nullptr) {
        ERROR("Set to not use dataset_broker and couldn't find ancestor of dataset {}. Make "
              "sure there is a stage upstream in the config, that adds dataset states. Unexpected "
              "nullptr: ",
              ds_id);
        if (!pstate)
            FATAL_ERROR("prodstate is a nullptr");
        if (!mstate)
            FATAL_ERROR("metadataState is a nullptr");
        if (!fstate)
            FATAL_ERROR("freqState is a nullptr");
    }

    // Get a reference to the acq state
    auto acq = acqs.at(ds_id);

    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        acq->freq_id_map[f.first] = ind++;
}

void VisWriter::write_data(Buffer* in_buf, int frame_id) {

    auto frame = VisFrameView(in_buf, frame_id);

    // Get time of the frame
    auto ftime = frame.time;
    time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};

    write_frame(frame, frame.dataset_id, frame.freq_id, t);
}
