
#include "VisWriter.hpp"

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "SystemInterface.hpp"   // for get_user_name, get_host_name
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "restServer.hpp"        // for connectionInstance
#include "version.h"             // for get_git_commit_hash
#include "visBuffer.hpp"         // for VisFrameView
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
    metadata["system_user"] = get_user_name();
    metadata["collection_server"] = get_host_name();

    return metadata;
}

/// Gets states from the dataset manager and saves some metadata
void VisWriter::get_dataset_state(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();

    // Get all states synchronously.
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, ds_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, ds_id);
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, ds_id);
    auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>, &dm, ds_id);

    const metadataState* mstate = mstate_fut.get();
    const stackState* sstate = sstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const prodState* pstate = pstate_fut.get();

    if (pstate == nullptr || mstate == nullptr || fstate == nullptr) {
        ERROR("Set to not use dataset_broker and couldn't find ancestor of dataset {}. Make "
              "sure there is a stage upstream in the config, that the dataset states. Unexpected "
              "nullptr: ",
              ds_id);
        if (!pstate)
            FATAL_ERROR("prodstate is a nullptr");
        if (!mstate)
            FATAL_ERROR("metadataState is a nullptr");
        if (!fstate)
            FATAL_ERROR("freqState is a nullptr");
    }

    {
        // Get a reference to the acq state
        auto acq = acqs.at(ds_id);

        uint ind = 0;
        for (auto& f : fstate->get_freqs())
            acq->freq_id_map[f.first] = ind++;

        acq->num_vis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();
    }
}

void VisWriter::write_data(Buffer* in_buf, int frame_id,
                           kotekan::prometheus::Gauge& write_time_metric) {

    const VisFrameView& frame = VisFrameView(in_buf, frame_id);

    dset_id_t dataset_id = frame.dataset_id;
    uint32_t freq_id = frame.freq_id;
    auto ftime = frame.time;

    // Check the dataset ID hasn't changed
    if (acqs.count(dataset_id) == 0) {
        init_acq(dataset_id);
    }

    // Get the acquisition we are writing into
    auto& acq = *(acqs.at(dataset_id));

    // If the dataset is bad, skip the frame and move onto the next
    if (acq.bad_dataset) {
        bad_dataset_frame_counter.labels({dataset_id.to_string()}).inc();

        // Check if the frequency we are receiving is on the list of frequencies
        // we are processing
        // TODO: this should probably be reported to prometheus
    } else if (acq.freq_id_map.count(freq_id) == 0) {
        WARN("Frequency id={:d} not enabled for VisWriter, discarding frame", freq_id);

        // Check that the number of visibilities matches what we expect
    } else if (frame.num_prod != acq.num_vis) {
        FATAL_ERROR("Number of products in frame doesn't match state or file ({:d} != {:d}).",
                    frame.num_prod, acq.num_vis);
        return;

    } else {

        // Get the time and frequency of the frame
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};
        uint32_t freq_ind = acq.freq_id_map.at(freq_id);

        // Add all the new information to the file.
        bool late;
        double start = current_time();

        // Write data
        late = acq.file_bundle->add_sample(t, freq_ind, frame);

        acq.last_update = current_time();
        double elapsed = acq.last_update - start;

        DEBUG("Written frequency {:d} in {:.5f} s", freq_id, elapsed);

        // Increase metric count if we dropped a frame at write time
        if (late) {
            late_frame_counter.labels({std::to_string(freq_id)}).inc();
        }

        // Update average write time in prometheus
        write_time.add_sample(elapsed);
        write_time_metric.set(write_time.average());
    }
}
