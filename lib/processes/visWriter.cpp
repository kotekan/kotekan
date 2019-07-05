#include "visWriter.hpp"

#include "StageFactory.hpp"
#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "version.h"
#include "visBuffer.hpp"
#include "visCompression.hpp"

#include "fmt.hpp"
#include "json.hpp"

#include <atomic>
#include <cxxabi.h>
#include <exception>
#include <functional>
#include <inttypes.h>
#include <regex>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <time.h>
#include <tuple>
#include <unistd.h>
#include <vector>


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(visWriter);
REGISTER_KOTEKAN_STAGE(visCalWriter);


// Define the string name of the bad frame types, required for the prometheus output
std::map<visWriter::droppedType, std::string> visWriter::dropped_type_map = {
    {visWriter::droppedType::late, "late"}, {visWriter::droppedType::bad_dataset, "bad_dataset"}};


visWriter::visWriter(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visWriter::main_thread, this)),
    dropped_frame_counter(Metrics::instance().add_counter("kotekan_viswriter_dropped_frame_total",
                                                          unique_name,
                                                          {"freq_id", "dataset_id", "reason"})) {

    // Fetch any simple configuration
    root_path = config.get_default<std::string>(unique_name, "root_path", ".");
    acq_timeout = config.get_default<double>(unique_name, "acq_timeout", 300);
    ignore_version = config.get_default<bool>(unique_name, "ignore_version", false);

    // Get the list of buffers that this stage shoud connect to
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Get the type of the file we are writing
    // TODO: we may want to validate here rather than at creation time
    file_type = config.get_default<std::string>(unique_name, "file_type", "hdf5fast");

    file_length = config.get_default<size_t>(unique_name, "file_length", 1024);
    window = config.get_default<size_t>(unique_name, "window", 20);

    // Check that the window isn't too long
    if (window > file_length) {
        std::string msg =
            fmt::format("Active times window ({}) should not be greater than file length"
                        " ({}). Setting window to file length.",
                        window, file_length);
        INFO(msg.c_str());
        window = file_length;
    }

    // TODO: get this from the datasetManager and put data type somewhere else
    instrument_name = config.get_default<std::string>(unique_name, "instrument_name", "chime");

    // Get the acq time out from the config
}

void visWriter::main_thread() {

    frameID frame_id(in_buf);

    auto& write_time_metric =
        Metrics::instance().add_gauge("kotekan_viswriter_write_time_seconds", unique_name);

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // Check the dataset ID hasn't changed
        if (acqs.count(frame.dataset_id) == 0) {
            string msg =
                fmt::format("Got new dataset ID={}. Starting a new acquisition.", frame.dataset_id);
            INFO(msg.c_str());

            init_acq(frame.dataset_id);
        }

        // Get the acquisition we are writing into
        auto& acq = acqs.at(frame.dataset_id);


        // If the dataset is bad, skip the frame and move onto the next
        if (acq.bad_dataset) {
            report_dropped_frame(frame.dataset_id, frame.freq_id, droppedType::bad_dataset);

            // Check if the frequency we are receiving is on the list of frequencies
            // we are processing
            // TODO: this should probably be reported to prometheus
        } else if (acq.freq_id_map.count(frame.freq_id) == 0) {
            WARN("Frequency id=%i not enabled for visWriter, discarding frame", frame.freq_id);

            // Check that the number of visibilities matches what we expect
        } else if (frame.num_prod != acq.num_vis) {
            string msg = fmt::format("Number of products in frame doesn't match state or file "
                                     "({} != {}).",
                                     frame.num_prod, acq.num_vis);
            ERROR(msg.c_str());
            raise(SIGINT);
            return;

        } else {

            // Get the time and frequency of the frame
            auto ftime = frame.time;
            time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};
            uint32_t freq_ind = acq.freq_id_map.at(frame.freq_id);

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

            DEBUG("Written frequency %i in %.5f s", frame.freq_id, elapsed);

            // Increase metric count if we dropped a frame at write time
            if (late) {
                report_dropped_frame(frame.dataset_id, frame.freq_id, droppedType::late);
            }

            // Update average write time in prometheus
            write_time.add_sample(elapsed);
            write_time_metric.set(write_time.average());
        }

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

        // Clean out any acquisitions that have been inactive long
        close_old_acqs();
    }
    Metrics::instance().remove_stage_metrics(unique_name);
}


void visWriter::report_dropped_frame(dset_id_t ds_id, uint32_t freq_id, droppedType reason) {

    // Get acqusition
    auto& acq = acqs.at(ds_id);

    // Relies on the fact that insertion zero intialises
    auto key = std::make_pair(freq_id, reason);
    // TODO: check if this is necessary
    acq.dropped_frame_count[key] += 1;
    dropped_frame_counter
        .labels({std::to_string(freq_id), std::to_string(ds_id), dropped_type_map.at(reason)})
        .inc();
}

void visWriter::close_old_acqs() {
    auto it = acqs.begin();
    while (it != acqs.end()) {
        double age = current_time() - it->second.last_update;

        if (!it->second.bad_dataset && age > acq_timeout) {
            it = acqs.erase(it);
        } else {
            it++;
        }
    }
}


void visWriter::get_dataset_state(dset_id_t ds_id) {

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
        ERROR("Set to not use dataset_broker and couldn't find "
              "ancestor of dataset 0x%" PRIx64 ". Make sure there is a stage"
              " upstream in the config, that the dataset states.\nExiting...",
              ds_id);
        ERROR("One of them is a nullptr (0): prodState %d, metadataState %d, "
              "freqState %d, stackState %d (but that one is okay).",
              pstate, mstate, fstate, sstate);
        raise(SIGINT);
    }

    // Get a reference to the acq state
    auto& acq = acqs.at(ds_id);

    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        acq.freq_id_map[f.first] = ind++;

    // compare git commit hashes
    // TODO: enforce and crash here if build type is Release?
    if (mstate->get_git_version_tag() != std::string(get_git_commit_hash())) {
        INFO("Git version tags don't match: dataset 0x%" PRIx64 " has tag %s,"
             "while the local git version tag is %s",
             ds_id, mstate->get_git_version_tag().c_str(), get_git_commit_hash());
    }

    acq.num_vis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();
}


bool visWriter::check_git_version(dset_id_t ds_id) {
    // Get the metadata state from the dM
    auto& dm = datasetManager::instance();
    const metadataState* mstate = dm.dataset_state<metadataState>(ds_id);

    // compare git commit hashes
    // TODO: enforce and crash here if build type is Release?
    if (mstate->get_git_version_tag() != std::string(get_git_commit_hash())) {
        WARN("Git version tags don't match: dataset 0x%" PRIx64 " has tag %s,"
             "while the local git version tag is %s",
             ds_id, mstate->get_git_version_tag().c_str(), get_git_commit_hash());
        return ignore_version;
    }
    return true;
}


void visWriter::init_acq(dset_id_t ds_id) {
    // Create the new acqState
    auto& acq = acqs[ds_id];

    // get dataset states
    get_dataset_state(ds_id);

    // Check the git version...
    if (!check_git_version(ds_id)) {
        acq.bad_dataset = true;
        return;
    }

    // TODO: chunk ID is not really supported now. Just set it to zero.
    uint32_t chunk_id = 0;

    // Construct metadata
    auto metadata = make_metadata(ds_id);

    try {
        acq.file_bundle =
            std::make_unique<visFileBundle>(file_type, root_path, instrument_name, metadata,
                                            chunk_id, file_length, window, ds_id, file_length);
    } catch (std::exception& e) {
        ERROR("Failed creating file bundle for new acquisition: %s", e.what());
        raise(SIGINT);
    }
}


std::map<std::string, std::string> visWriter::make_metadata(dset_id_t ds_id) {
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


visCalWriter::visCalWriter(Config& config, const string& unique_name,
                           bufferContainer& buffer_container) :
    visWriter::visWriter(config, unique_name, buffer_container) {

    // Register REST callback
    endpoint = "/release_live_file/" + std::regex_replace(unique_name, std::regex("^/+"), "");
    using namespace std::placeholders;
    restServer::instance().register_get_callback(endpoint,
                                                 std::bind(&visCalWriter::rest_callback, this, _1));

    // Get file name to write to
    // TODO: strip file extensions?
    std::string fname_base = config.get_default<std::string>(unique_name, "file_base", "cal");
    acq_name = config.get_default<std::string>(unique_name, "dir_name", "cal");
    // Initially start with this buffer configuration
    fname_live = fname_base + "_A";
    fname_frozen = fname_base + "_B";

    // Use a very short window by default
    window = config.get_default<size_t>(unique_name, "window", 10);

    // Force use of VisFileRing
    file_type = "ring";

    // Check if any of these files exist
    std::string full_path = root_path + "/" + acq_name + "/";
    if ((access((full_path + fname_base + "_A.data").c_str(), F_OK) == 0)
        || (access((full_path + fname_base + "_B.data").c_str(), F_OK) == 0)) {
        INFO(("Clobering files in " + full_path).c_str());
        check_remove(full_path + fname_base + "_A.data");
        check_remove("." + full_path + fname_base + "_A.lock");
        check_remove(full_path + fname_base + "_A.meta");
        check_remove(full_path + fname_base + "_B.data");
        check_remove("." + full_path + fname_base + "_B.lock");
        check_remove(full_path + fname_base + "_B.meta");
    }
}

visCalWriter::~visCalWriter() {
    restServer::instance().remove_get_callback(endpoint);
    Metrics::instance().remove_stage_metrics(unique_name);
}

void visCalWriter::rest_callback(connectionInstance& conn) {
    // Ensure no write is ongoing
    std::lock_guard<std::mutex> write_guard(write_mutex);

    INFO("Received request to release calibration live file...");

    // Swap files
    std::string fname_tmp = fname_live;
    fname_live = fname_frozen;
    fname_frozen = fname_tmp;

    // Tell visCalFileBundle to write to new file starting with next sample
    file_cal_bundle->swap_file(fname_live, acq_name);

    // Respond with frozen file path
    json reply{"file_path", root_path + "/" + acq_name + "/" + fname_frozen};
    conn.send_json_reply(reply);
    INFO("Done. Resuming write loop.");
}


void visCalWriter::init_acq(dset_id_t ds_id) {
    // Count the number of enabled acqusitions, for the visCalWriter this can't
    // be more than one
    int num_enabled = std::count_if(acqs.begin(), acqs.end(),
                                    [](auto& item) -> bool { return !(item.second.bad_dataset); });

    // Create the new acqState
    auto& acq = acqs[ds_id];

    // get dataset states
    get_dataset_state(ds_id);

    // Check there are no other valid acqs
    if (num_enabled > 0) {
        WARN("visCalWriter can only have one acquistion. Dropping all frames "
             "of dataset_id=%" PRIx64,
             ds_id);
        acq.bad_dataset = true;
        return;
    }

    // Check the git version...
    if (!check_git_version(ds_id)) {
        acq.bad_dataset = true;
        return;
    }

    // TODO: chunk ID is not really supported now. Just set it to zero.
    uint32_t chunk_id = 0;

    // Construct metadata
    auto metadata = make_metadata(ds_id);

    // Create the derived class visCalFileBundle, and save to the instance, then
    // return as a unique_ptr
    file_cal_bundle = new visCalFileBundle(file_type, root_path, instrument_name, metadata,
                                           chunk_id, file_length, window, ds_id, file_length);
    file_cal_bundle->set_file_name(fname_live, acq_name);

    acq.file_bundle = std::unique_ptr<visFileBundle>(file_cal_bundle);
}
