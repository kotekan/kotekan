
#include "HFBWriter.hpp"

#include "Config.hpp"            // for Config
#include "HFBFrameView.hpp"      // for HFBFrameView
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
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

REGISTER_KOTEKAN_STAGE(HFBWriter);

HFBWriter::HFBWriter(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container) :
    BaseWriter(config, unique_name, buffer_container){};

/// Construct the set of metadata
std::map<std::string, std::string> HFBWriter::make_metadata(dset_id_t ds_id) {

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
    metadata["instrument_name"] = mstate->get_instrument_name();
    metadata["notes"] = ""; // TODO: connect up notes
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = user;
    metadata["collection_server"] = hostname;
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

    {
        // std::lock_guard<std::mutex> _lock(acqs_mutex);
        // Get a reference to the acq state
        auto acq = acqs.at(ds_id);

        uint ind = 0;
        for (auto& f : fstate->get_freqs())
            acq->freq_id_map[f.first] = ind++;

        acq->num_beams = bstate->get_beams().size();
    }
}

void HFBWriter::write_data(Buffer* in_buf, int frame_id,
                           kotekan::prometheus::Gauge& write_time_metric,
                           std::unique_lock<std::mutex>& acqs_lock) {

    const HFBFrameView& frame = HFBFrameView(in_buf, frame_id);

    dset_id_t dataset_id = frame.dataset_id;
    uint32_t freq_id = frame.freq_id;
    auto time = frame.time;
    uint64_t fpga_seq_num = frame.fpga_seq_num;

    acqs_lock.lock();
    // Check the dataset ID hasn't changed
    if (acqs.count(dataset_id) == 0) {
        init_acq(dataset_id);
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
        WARN("Frequency id={:d} not enabled for Writer, discarding frame", freq_id);

        // Check that the number of visibilities matches what we expect
    } else if (frame.num_beams != acq.num_beams) {
        FATAL_ERROR("Number of beams in frame doesn't match state or file ({:d} != {:d}).",
                    frame.num_beams, acq.num_beams);
        return;

    } else {

        // Get the time and frequency of the frame

        time_ctype t = {fpga_seq_num, ts_to_double(time)};
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
