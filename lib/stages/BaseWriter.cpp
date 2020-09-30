#include "BaseWriter.hpp"

#include "Config.hpp"            // for Config
#include "FrameView.hpp"         // for FrameView
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
#include "visFile.hpp"           // for visFileBundle, _factory_aliasvisFile

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


BaseWriter::BaseWriter(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&BaseWriter::main_thread, this)),
    late_frame_counter(Metrics::instance().add_counter("kotekan_writer_late_frame_total",
                                                       unique_name, {"freq_id"})),
    bad_dataset_frame_counter(Metrics::instance().add_counter(
        "kotekan_writer_bad_dataset_frame_total", unique_name, {"dataset_id"})),
    write_time_metric(
        Metrics::instance().add_gauge("kotekan_writer_write_time_seconds", unique_name)) {

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

void BaseWriter::main_thread() {

    frameID frame_id(in_buf);

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), frame_id) == nullptr) {
            break;
        }

        // Write frame
        write_data(in_buf, frame_id);

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);

        // Clean out any acquisitions that have been inactive long
        close_old_acqs();
    }
}

void BaseWriter::init_acq(dset_id_t ds_id) {

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

    // Construct metadata
    auto metadata = make_metadata(ds_id);

    try {
        acq.file_bundle = std::make_unique<visFileBundle>(
            file_type, root_path, instrument_name, metadata, chunk_id, file_length, window,
            kotekan::logLevel(_member_log_level), ds_id, file_length);
    } catch (std::exception& e) {
        FATAL_ERROR("Failed creating file bundle for new acquisition: {:s}", e.what());
    }
}

void BaseWriter::write_frame(const FrameView& frame, dset_id_t dataset_id, uint32_t freq_id,
                             time_ctype time, size_t frame_size) {

    static bool first = true;

    // Check the dataset ID hasn't changed
    if (acqs.count(dataset_id) == 0) {
        init_acq(dataset_id);
    }

    // Get the acquisition we are writing into
    auto& acq = *(acqs.at(dataset_id));

    // Store the initial frame size to check future frame sizes against
    if (first) {
        init_frame_size = frame_size;
        first = false;
    }

    // If the dataset is bad, skip the frame and move onto the next
    if (acq.bad_dataset) {
        bad_dataset_frame_counter.labels({dataset_id.to_string()}).inc();

        // Check if the frequency we are receiving is on the list of frequencies
        // we are processing
        // TODO: this should probably be reported to prometheus
    } else if (acq.freq_id_map.count(freq_id) == 0) {
        WARN("Frequency id={:d} not enabled for Writer, discarding frame", freq_id);

        // Check that the frame size matches what we expect
    } else if (frame_size != init_frame_size) {
        FATAL_ERROR("Size of frame doesn't match first frame ({:d} != {:d}).", frame_size,
                    init_frame_size);
        return;

    } else {

        // Get frequency of the frame
        uint32_t freq_ind = acq.freq_id_map.at(freq_id);

        // Add all the new information to the file.
        bool late;
        double start = current_time();

        // Write data
        late = acq.file_bundle->add_sample(time, freq_ind, frame);

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

void BaseWriter::close_old_acqs() {

    // Sweep over all both acq storing maps and delete any entries for expired
    // acquisitions. Must loop over both to actually remove and close the
    // acquisitions

    /// Only sweep over the acqs to decide which to close every 1/3 of the
    /// acq_timeout
    double now = current_time();
    if (now > next_sweep) {
        next_sweep = now + acq_timeout / 3.0;
    }

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

bool BaseWriter::check_git_version(dset_id_t ds_id) {
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
