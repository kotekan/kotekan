#include "Writer.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator<
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for dset_id_t, fingerprint_t, datasetManager
#include "datasetState.hpp"      // for metadataState, freqState, prodState, stackState, _facto...
#include "factory.hpp"           // for FACTORY
#include "kotekanLogging.hpp"    // for INFO, ERROR, WARN, FATAL_ERROR, DEBUG, logLevel
#include "prometheusMetrics.hpp" // for Counter, Metrics, MetricFamily, Gauge
#include "restServer.hpp"        // for restServer, connectionInstance, HTTP_RESPONSE
#include "version.h"             // for get_git_commit_hash
#include "visBuffer.hpp"         // for VisFrameView
#include "HfbFrameView.hpp"      // for HfbFrameView
#include "visFile.hpp"           // for visFileBundle, visCalFileBundle, _factory_aliasvisFile

#include "fmt.hpp"  // for format, fmt
#include "json.hpp" // for json_ref, json

#include <algorithm>    // for copy, copy_backward, count_if, equal, max
#include <atomic>       // for atomic_bool
#include <cxxabi.h>     // for __forced_unwind
#include <deque>        // for deque
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, _Placeholder, bind, _1, function
#include <regex>        // for match_results<>::_Base_type, regex_replace, regex
#include <sstream>      // for basic_stringbuf<>::int_type, basic_stringbuf<>::pos_type
#include <sys/types.h>  // for uint
#include <system_error> // for system_error
#include <tuple>        // for get
#include <vector>       // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(Writer);
REGISTER_KOTEKAN_STAGE(visCalWriter);


Writer::Writer(Config& config, const std::string& unique_name,
                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&Writer::main_thread, this)),
    late_frame_counter(Metrics::instance().add_counter("kotekan_writer_late_frame_total",
                                                       unique_name, {"freq_id"})),
    bad_dataset_frame_counter(Metrics::instance().add_counter(
        "kotekan_writer_bad_dataset_frame_total", unique_name, {"dataset_id"})) {

    // Fetch any simple configuration
    root_path = config.get_default<std::string>(unique_name, "root_path", ".");
    acq_timeout = config.get_default<double>(unique_name, "acq_timeout", 300);
    ignore_version = config.get_default<bool>(unique_name, "ignore_version", false);

    // Get the type of the file we are writing
    // TODO: we may want to validate here rather than at creation time
    file_type = config.get_default<std::string>(unique_name, "file_type", "hdf5fast");
    if (!FACTORY(visFile)::exists(file_type)) {
        FATAL_ERROR("Unknown file type '{}'", file_type);
        return;
    }

    file_length = config.get_default<size_t>(unique_name, "file_length", 1024);
    window = config.get_default<size_t>(unique_name, "window", 20);

    // Check that the window isn't too long
    if (window > file_length) {
        INFO("Active times window ({:d}) should not be greater than file length ({:d}). Setting "
             "window to file length.",
             window, file_length);
        window = file_length;
    }

    // TODO: get this from the datasetManager and put data type somewhere else
    instrument_name = config.get_default<std::string>(unique_name, "instrument_name", "chime");

    // Set the list of critical states
    critical_state_types = {"frequencies", "inputs",      "products",
                            "stack",       "eigenvalues", "metadata"};
    auto t = config.get_default<std::vector<std::string>>(unique_name, "critical_states", {});
    for (const auto& state : t) {
        if (!FACTORY(datasetState)::exists(state)) {
            FATAL_ERROR("Unknown datasetState type '{}' given as `critical_state`", state);
            return;
        }
        critical_state_types.insert(state);
    }

    // Get the list of buffers that this stage should connect to
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

void Writer::main_thread() {

    frameID frame_id(in_buf);

    auto& write_time_metric =
        Metrics::instance().add_gauge("kotekan_writer_write_time_seconds", unique_name);

    std::unique_lock<std::mutex> acqs_lock(acqs_mutex, std::defer_lock);

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        if(!strcmp(in_buf->buffer_type, "vis")) {
          auto frame = VisFrameView(in_buf, frame_id);
          write_vis_data(frame, write_time_metric, acqs_lock);
        }
        else if(!strcmp(in_buf->buffer_type, "hfb")) {
          auto frame = HfbFrameView(in_buf, frame_id);
          write_hfb_data(frame, write_time_metric, acqs_lock);
        }

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

        // Clean out any acquisitions that have been inactive long
        close_old_acqs();
    }
}

void Writer::write_vis_data(VisFrameView frame, auto& write_time_metric, std::unique_lock<std::mutex>& acqs_lock) {

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
      WARN("Frequency id={:d} not enabled for Writer, discarding frame", freq_id);

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

void Writer::write_hfb_data(HfbFrameView frame, auto& write_time_metric, std::unique_lock<std::mutex>& acqs_lock) {

    dset_id_t dataset_id = frame.dataset_id;
    uint32_t freq_id = frame.freq_id;
    auto time = frame.time;
    uint64_t fpga_seq_num = frame.fpga_seq_num;
  
    acqs_lock.lock();
    // Check the dataset ID hasn't changed
    if (acqs.count(dataset_id) == 0) {
      init_acq(dataset_id, make_hfb_metadata());
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

void Writer::close_old_acqs() {

    // Sweep over all both acq storing maps and delete any entries for expired
    // acquisitions. Must loop over both to actually remove and close the
    // acquisitions

    /// Only sweep over the acqs to decide which to close every 1/3 of the
    /// acq_timeout
    double now = current_time();
    if (now > next_sweep) {
        next_sweep = now + acq_timeout / 3.0;
    }

    {
        std::lock_guard<std::mutex> _lock(acqs_mutex);
        // Scan over the dataset keyed list
        auto it1 = acqs.begin();
        while (it1 != acqs.end()) {
            double age = now - it1->second->last_update;

            if (!it1->second->bad_dataset && age > acq_timeout) {
                it1 = acqs.erase(it1);
            } else {
                it1++;
            }
        }

        // Scan over the fingerprint keyed list
        auto it2 = acqs_fingerprint.begin();
        while (it2 != acqs_fingerprint.end()) {
            double age = now - it2->second->last_update;

            if (!it2->second->bad_dataset && age > acq_timeout) {
                it2 = acqs_fingerprint.erase(it2);
            } else {
                it2++;
            }
        }
    }
}


void Writer::get_dataset_state(dset_id_t ds_id) {

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
        acq->num_beams = config.get_default<uint32_t>(unique_name, "num_frb_total_beams", 1024);
    }
}


bool Writer::check_git_version(dset_id_t ds_id) {
    // Get the metadata state from the dM
    auto& dm = datasetManager::instance();
    const metadataState* mstate = dm.dataset_state<metadataState>(ds_id);

    // compare git commit hashes
    // TODO: enforce and crash here if build type is Release?
    if (mstate->get_git_version_tag() != std::string(get_git_commit_hash())) {
        if (ignore_version) {
            WARN("Git version tags don't match: dataset {} has tag {:s},"
                 "while the local git version tag is {:s}. Ignoring for now, but you should look "
                 "into this.",
                 ds_id, mstate->get_git_version_tag(), get_git_commit_hash());
            return true;
        } else {
            WARN("Git version tags don't match: dataset {} has tag {:s},"
                 "while the local git version tag is {:s}. Marking acqusition bad and dropping all "
                 "data.",
                 ds_id, mstate->get_git_version_tag(), get_git_commit_hash());
            return false;
        }
    }
    return true;
}


void Writer::init_acq(dset_id_t ds_id, std::map<std::string, std::string> metadata) {

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


std::map<std::string, std::string> Writer::make_metadata(dset_id_t ds_id) {
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
    metadata["num_beams"] = std::to_string(config.get<uint32_t>(unique_name, "num_frb_total_beams"));
    metadata["num_subfreq"] = std::to_string(config.get<uint32_t>(unique_name, "num_sub_freqs"));

    return metadata;
}

std::map<std::string, std::string> Writer::make_hfb_metadata() {

    // Get the current user
    std::string user(256, '\0');
    user = (getlogin_r(&user[0], 256) == 0) ? user.c_str() : "unknown";

    // Get the current hostname of the system for the metadata
    std::string hostname(256, '\0');
    gethostname(&hostname[0], 256);
    hostname = hostname.c_str();

    // Set the metadata that we want to save with the file
    std::map<std::string, std::string> metadata;
    metadata["instrument_name"] = instrument_name;
    metadata["notes"] = ""; // TODO: connect up notes
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = user;
    metadata["collection_server"] = hostname;
    metadata["num_beams"] = std::to_string(config.get<uint32_t>(unique_name, "num_frb_total_beams"));
    metadata["num_sub_freqs"] = std::to_string(config.get<uint32_t>(unique_name, "num_sub_freqs"));

    return metadata;
}


visCalWriter::visCalWriter(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Writer::Writer(config, unique_name, buffer_container) {

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
    DEBUG("Checking for and removing old buffer files...");
    std::string full_path = root_path + "/" + acq_name + "/";
    check_remove(full_path + fname_base + "_A.data");
    check_remove(full_path + "." + fname_base + "_A.lock");
    check_remove(full_path + fname_base + "_A.meta");
    check_remove(full_path + fname_base + "_B.data");
    check_remove(full_path + "." + fname_base + "_B.lock");
    check_remove(full_path + fname_base + "_B.meta");

    file_cal_bundle = nullptr;

    // Register REST callback (needs to be done at the end such that the names have been set)
    using namespace std::placeholders;
    endpoint = "/release_live_file/" + std::regex_replace(unique_name, std::regex("^/+"), "");
    restServer::instance().register_get_callback(endpoint,
                                                 std::bind(&visCalWriter::rest_callback, this, _1));
}

visCalWriter::~visCalWriter() {
    restServer::instance().remove_get_callback(endpoint);
}

void visCalWriter::rest_callback(connectionInstance& conn) {

    INFO("Received request to release calibration live file...");

    // Need to deal with the case that we could release the data before any acq has been started
    if (acqs_fingerprint.size() == 0 || file_cal_bundle == nullptr) {
        WARN("Asked to release calibration, but not active data.");
        return;
    }

    // Swap files
    std::string fname_tmp = fname_live;
    fname_live = fname_frozen;
    fname_frozen = fname_tmp;

    // Tell visCalFileBundle to write to new file starting with next sample
    {
        // Ensure no write is ongoing
        std::lock_guard<std::mutex> write_guard(write_mutex);
        file_cal_bundle->swap_file(fname_live, acq_name);
    }

    // Respond with frozen file path
    nlohmann::json reply{"file_path",
                         fmt::format(fmt("{:s}/{:s}/{:s}"), root_path, acq_name, fname_frozen)};
    conn.send_json_reply(reply);
    INFO("Done. Resuming write loop.");
}


void visCalWriter::init_acq(dset_id_t ds_id, std::map<std::string, std::string> metadata) {

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

    INFO("Got new dataset ID={} with new fingerprint={}.", ds_id, fp);

    // Count the number of enabled acqusitions, for the visCalWriter this can't
    // be more than one
    int num_enabled = std::count_if(acqs_fingerprint.begin(), acqs_fingerprint.end(),
                                    [](auto& item) -> bool { return !(item.second->bad_dataset); });

    // If it is not known we need to initialise everything
    acqs_fingerprint[fp] = std::make_shared<acqState>();
    acqs[ds_id] = acqs_fingerprint.at(fp);
    auto& acq = *(acqs.at(ds_id));

    // get dataset states
    get_dataset_state(ds_id);

    // Check there are no other valid acqs
    if (num_enabled > 0) {
        WARN("visCalWriter can only have one acquistion. Dropping all frames of dataset_id {}",
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

    // Create the derived class visCalFileBundle, and save to the instance, then
    // return as a unique_ptr
    file_cal_bundle =
        new visCalFileBundle(file_type, root_path, instrument_name, metadata, chunk_id, file_length,
                             window, kotekan::logLevel(_member_log_level), ds_id, file_length);
    file_cal_bundle->set_file_name(fname_live, acq_name);

    acq.file_bundle = std::unique_ptr<visFileBundle>(file_cal_bundle);
}
