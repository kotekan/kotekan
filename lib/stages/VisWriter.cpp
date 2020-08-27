
#include "VisWriter.hpp"

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t
#include "prometheusMetrics.hpp" // for Counter, MetricFamily
#include "restServer.hpp"        // for connectionInstance
#include "version.h"             // for get_git_commit_hash
#include "visBuffer.hpp"         // for VisFrameView
#include "visFile.hpp"           // for visFileBundle, visCalFileBundle
#include "visUtil.hpp"           // for movingAverage

#include <cstdint>   // for uint32_t
#include <errno.h>   // for ENOENT, errno
#include <future>    // for future
#include <map>       // for map
#include <memory>    // for shared_ptr, unique_ptr
#include <mutex>     // for mutex
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

void VisWriter::main_thread() {

    frameID frame_id(in_buf);

    kotekan::prometheus::Gauge& write_time_metric =
        Metrics::instance().add_gauge("kotekan_writer_write_time_seconds", unique_name);

    std::unique_lock<std::mutex> acqs_lock(acqs_mutex, std::defer_lock);

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = VisFrameView(in_buf, frame_id);
        write_data(frame, write_time_metric, acqs_lock);

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

        // Clean out any acquisitions that have been inactive long
        close_old_acqs();
    }
}
/// Construct the set of metadata
std::map<std::string, std::string> VisWriter::make_metadata(dset_id_t ds_id) {

    // Get the metadata state from the dM
    auto& dm = datasetManager::instance();
    const metadataState* mstate = dm.dataset_state<metadataState>(ds_id);

    // Get the current user
    std::string user(256, '\0');
    user = (getlogin_r(&user[0], 256) == 0) ? user.c_str() : "unknown";

    // Get the current hostname of the system for the metadata
    std::string hostname(256, '\0');
    gethostname(&hostname[0], 256);
    hostname = hostname.c_str();

    // Set the metadata that we want to save with the file
    std::map<std::string, std::string> metadata;
    metadata["weight_type"] = mstate->get_weight_type();
    metadata["archive_version"] = "NT_3.1.0";
    metadata["instrument_name"] = instrument_name;
    metadata["notes"] = ""; // TODO: connect up notes
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = user;
    metadata["collection_server"] = hostname;

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
        // std::lock_guard<std::mutex> _lock(acqs_mutex);
        // Get a reference to the acq state
        auto acq = acqs.at(ds_id);

        uint ind = 0;
        for (auto& f : fstate->get_freqs())
            acq->freq_id_map[f.first] = ind++;

        acq->num_vis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();
    }
}

void VisWriter::write_data(const FrameView& frame_view,
                           kotekan::prometheus::Gauge& write_time_metric,
                           std::unique_lock<std::mutex>& acqs_lock) {

    const VisFrameView& frame = static_cast<const VisFrameView&>(frame_view);

    dset_id_t dataset_id = frame.dataset_id;
    uint32_t freq_id = frame.freq_id;
    auto ftime = frame.time;

    acqs_lock.lock();
    // Check the dataset ID hasn't changed
    if (acqs.count(dataset_id) == 0) {
        init_acq(dataset_id, make_metadata(dataset_id));
    }

    // Get the acquisition we are writing into
    auto& acq = *(acqs.at(dataset_id));
    acqs_lock.unlock();

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

        // Lock and write data
        {
            std::lock_guard<std::mutex> lock(write_mutex);
            late = acq.file_bundle->add_sample(t, freq_ind, frame);
        }
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

void VisWriter::init_acq(dset_id_t ds_id, std::map<std::string, std::string> metadata) {

    // Create the new acqState
    auto fp = datasetManager::instance().fingerprint(ds_id, critical_state_types);

    // If the fingerprint is already known, we don't need to start a new
    // acquisition, just add a map from the dataset_id to the acquisition we
    // should use.
    if (acqs_fingerprint.count(fp) > 0) {
        INFO("Got new dataset ID={} with known fingerprint={}.", ds_id, fp);
        acqs[ds_id] = acqs_fingerprint.at(fp);
        return;
    }

    INFO("Got new dataset ID={} with new fingerprint={}. Creating new acquisition.", ds_id, fp);

    // If it is not known we need to initialise everything
    acqs_fingerprint[fp] = std::make_shared<acqState>();
    acqs[ds_id] = acqs_fingerprint.at(fp);
    auto& acq = *(acqs.at(ds_id));

    // get dataset states
    get_dataset_state(ds_id);

    // Check the git version...
    if (!check_git_version(ds_id)) {
        acq.bad_dataset = true;
        return;
    }

    // TODO: chunk ID is not really supported now. Just set it to zero.
    uint32_t chunk_id = 0;

    try {
        acq.file_bundle = std::make_unique<visFileBundle>(
            file_type, root_path, instrument_name, metadata, chunk_id, file_length, window,
            kotekan::logLevel(_member_log_level), ds_id, file_length);
    } catch (std::exception& e) {
        FATAL_ERROR("Failed creating file bundle for new acquisition: {:s}", e.what());
    }
}
