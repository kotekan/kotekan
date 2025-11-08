#include "VisWriter.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "SystemInterface.hpp" // for get_hostname, get_username
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for datasetManager
#include "datasetState.hpp"    // for freqState, metadataState
#include "kotekanLogging.hpp"  // for FATAL_ERROR
#include "version.h"           // for get_git_commit_hash
#include "visBuffer.hpp"       // for VisFrameView
#include "visUtil.hpp"         // for ts_to_double, time_ctype

#include <future>  // for async
#include <utility> // for std::get

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_KOTEKAN_STAGE(VisWriter);

VisWriter::VisWriter(Config& config, const std::string& unique_name,
                     bufferContainer& buffer_container) :
    BaseWriter(config, unique_name, buffer_container) {
    notes = config.get_default<std::string>(unique_name, "notes", "");
    acq_fmt = "{acq_start:%Y%m%dT%H%M%SZ}_" + instrument_name + "_vis";
    file_fmt = "vis_{seconds_since_start:08d}_0000";
}

std::map<std::string, std::string> VisWriter::make_metadata(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();
    const metadataState* mstate = dm.dataset_state<metadataState>(ds_id);

    if (mstate == nullptr) {
        FATAL_ERROR("Set to not use dataset_broker and couldn't find metadataState ancestor of "
                    "dataset {}.",
                    ds_id);
    }

    std::map<std::string, std::string> metadata;
    metadata["weight_type"] = mstate->get_weight_type();
    metadata["instrument_name"] = mstate->get_instrument_name();
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["notes"] = notes;
    metadata["system_user"] = get_username();
    metadata["collection_server"] = get_hostname();

    return metadata;
}

void VisWriter::get_dataset_state(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    const freqState* fstate = fstate_fut.get();

    if (fstate == nullptr) {
        FATAL_ERROR("Set to not use dataset_broker and couldn't find freqState ancestor of dataset "
                    "{}.",
                    ds_id);
        return;
    }

    auto acq = acqs.at(ds_id);
    acq->freq_id_map.clear();
    uint32_t ind = 0;
    for (auto& f : fstate->get_freqs()) {
        acq->freq_id_map[f.first] = ind++;
    }
}

void VisWriter::write_data(Buffer* in_buf, int frame_id) {

    VisFrameView frame(in_buf, frame_id);
    auto frame_time = frame.time;
    time_ctype t = {std::get<0>(frame_time), ts_to_double(std::get<1>(frame_time))};
    write_frame(frame, frame.dataset_id, frame.freq_id, t);
}
