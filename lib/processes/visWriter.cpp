#include "visWriter.hpp"

#include <cxxabi.h>
#include <signal.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <atomic>
#include <exception>
#include <functional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <inttypes.h>

#include "fmt.hpp"
#include "json.hpp"

#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "processFactory.hpp"
#include "prometheusMetrics.hpp"
#include "version.h"
#include "visBuffer.hpp"
#include "visCompression.hpp"


REGISTER_KOTEKAN_PROCESS(visWriter);
REGISTER_KOTEKAN_PROCESS(visCalWriter);

visWriter::visWriter(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visWriter::main_thread, this)) {

    // Fetch any simple configuration
    root_path = config.get_default<std::string>(unique_name, "root_path", ".");

    // Get the list of buffers that this process shoud connect to
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Get the type of the file we are writing
    // TODO: we may want to validate here rather than at creation time
    file_type = config.get_default<std::string>(
                unique_name, "file_type", "hdf5fast");

    file_length = config.get_default<size_t>(unique_name, "file_length", 1024);
    window = config.get_default<size_t>(unique_name, "window", 20);

    // TODO: get this from the datasetManager and put data type somewhere else
    instrument_name = config.get_default<std::string>(
                unique_name, "instrument_name", "chime");

    // Set the instrument name from the hostname if in node mode
    node_mode = config.get_default<bool>(unique_name, "node_mode", false);
    if(node_mode) {
        std::string t(256, '\0');
        gethostname(&t[0], 256);
        // Here we trim the hostname to the first alphanumeric segment only.
        instrument_name = t.substr(0, (t + ".").find_first_of(".-"));
    }
}

void visWriter::main_thread() {

    frameID frame_id(in_buf);

    // Wait for the first frame to get the dataset ID
    if (wait_for_full_frame(in_buf, unique_name.c_str(), 0) == nullptr)
        return;
    auto frame = visFrameView(in_buf, 0);

    // Get the dataset ID that the downstream files can use to determine what
    // they will be writing.
    ds_id = frame.dataset_id;

    // Setup the acquisition, create a new file
    auto init_fut = std::async(&visWriter::init_acq, this);

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // Check the dataset ID hasn't changed
        if (frame.dataset_id != ds_id) {
            string msg = fmt::format(
                        "Unexpected dataset ID={} received (expected id={}). " \
                        "Creating a new file.",
                frame.dataset_id, ds_id
            );
            INFO(msg.c_str());
            auto init_fut = std::async(&visWriter::init_acq, this);
        }

        // Construct the new time
        auto ftime = frame.time;
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};


        // Are we waiting for datasetStates? We need the states now...wait here.
        if (init_fut.valid()) {
            // Get void future, otherwise it will stay valid.
            init_fut.get();
        }

        // Check if the frequency we are receiving is on the list of frequencies
        // we are processing
        if (_freq_id_map.count(frame.freq_id) == 0) {
            WARN("Frequency id=%i not enabled for visWriter, discarding frame",
                 frame.freq_id);

        // Check that the number of visibilities matches what we expect
        } else if (frame.num_prod != _num_vis) {
            string msg = fmt::format(
                "Number of products in frame doesn't match state or file " \
                        "({} != {}).",
                frame.num_prod, _num_vis
            );
            ERROR(msg.c_str());
            raise(SIGINT);
            return;

        } else {

            DEBUG("Writing frequency id=%i", frame.freq_id);

            uint32_t freq_ind = _freq_id_map[frame.freq_id];

            // Add all the new information to the file.
            double start = current_time();
            write_mutex.lock();
            bool error = file_bundle->add_sample(t, freq_ind, frame);
            write_mutex.unlock();
            double elapsed = current_time() - start;

            DEBUG("Write time %.5f s", elapsed);

            // Increase metric count if we dropped a frame at write time
            if(error) {
                auto key = std::make_pair(frame.dataset_id, frame.freq_id);
                // Relies on the fact that insertion zero intialises
                dropped_frame_count[key] += 1;
                std::string labels = fmt::format("freq_id=\"{}\"," \
                                                 "dataset_id=\"{}\"",
                                                 frame.freq_id,
                                                 frame.dataset_id);
                prometheusMetrics::instance().add_process_metric(
                            "kotekan_viswriter_dropped_frame_total",
                            unique_name,
                            dropped_frame_count.at(key),
                            labels);
            }

            // Update average write time in prometheus
            write_time.add_sample(elapsed);
            prometheusMetrics::instance().add_process_metric(
                "kotekan_viswriter_write_time_seconds",
                unique_name, write_time.average()
            );

        }

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id++);
    }
}

void visWriter::get_dataset_state() {

    auto& dm = datasetManager::instance();

    // Get all states synchronously.
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm,
                                 ds_id);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm,
                                 ds_id);
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>,
                                 &dm, ds_id);
    auto mstate_fut = std::async(&datasetManager::dataset_state<metadataState>,
                                 &dm, ds_id);

    const metadataState* mstate = mstate_fut.get();
    const stackState* sstate = sstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const prodState* pstate = pstate_fut.get();

    if (pstate == nullptr || mstate == nullptr || fstate == nullptr) {
        ERROR("Set to not use dataset_broker and couldn't find " \
              "ancestor of dataset 0x%" PRIx64 ". Make sure there is a process"\
              " upstream in the config, that the dataset states.\nExiting...",
              ds_id);
        ERROR("One of them is a nullptr (0): prodState %d, metadataState %d, " \
              "freqState %d, stackState %d (but that one is okay).", pstate,
              mstate, fstate, sstate);
        raise(SIGINT);
    }

    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        _freq_id_map[f.first] = ind++;

    weights_type = mstate->get_weight_type();

    // compare git commit hashes
    // TODO: enforce and crash here if build type is Release?
    if (mstate->get_git_version_tag() != std::string(get_git_commit_hash())) {
        INFO("Git version tags don't match: dataset 0x%" PRIx64 " has tag %s," \
             "while the local git version tag is %s", ds_id,
             mstate->get_git_version_tag().c_str(),
             get_git_commit_hash());
    }

    _num_vis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();
}

void visWriter::init_acq() {

    // get dataset states
    get_dataset_state();

    // TODO: chunk ID is not really supported now. Just set it to zero.
    chunk_id = 0;

    // Get the current user
    std::string user(256, '\0');
    user = (getlogin_r(&user[0], 256) == 0) ? user.c_str() : "unknown";

    // Get the current hostname of the system for the metadata
    std::string hostname(256, '\0');
    gethostname(&hostname[0], 256);
    hostname = hostname.c_str();

    // Set the metadata that we want to save with the file
    std::map<std::string, std::string> metadata;
    metadata["weight_type"] = weights_type;
    metadata["archive_version"] = "NT_3.1.0";
    metadata["instrument_name"] = instrument_name;
    metadata["notes"] = "";   // TODO: connect up notes
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = user;
    metadata["collection_server"] = hostname;

    // For a ring-type file, rollover must be disabled,
    // window should be less than file length
    rollover = file_length;
    if (file_type == "ring") {
        rollover = 0;
        if (window > file_length) {
            INFO("Active times window cannot be greater than file length for "
                 "ring type files. Setting window to file length.");
            window = file_length;
        }
    }
    make_bundle(metadata);
}


void visWriter::make_bundle(std::map<std::string, std::string>& metadata) {
    file_bundle = std::make_unique<visFileBundle>(
        file_type, root_path, instrument_name, metadata, chunk_id,
        rollover, window, ds_id, file_length
    );
}


visCalWriter::visCalWriter(Config &config,
                            const string& unique_name,
                            bufferContainer &buffer_container) :
    visWriter::visWriter(config, unique_name, buffer_container) {

    // Register REST callback
    endpoint = "/release_live_file/" + std::regex_replace(unique_name, std::regex("^/+"), "");
    using namespace std::placeholders;
    restServer::instance().register_get_callback(endpoint,
            std::bind(&visCalWriter::rest_callback, this, _1));

    // Get file name to write to
    // TODO: strip file extensions?
    std::string fname_base = config.get_default<std::string>(
                unique_name, "file_base", "cal");
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
        check_remove((full_path + fname_base + "_A.data").c_str());
        check_remove(("." + full_path + fname_base + "_A.lock").c_str());
        check_remove((full_path + fname_base + "_A.meta").c_str());
        check_remove((full_path + fname_base + "_B.data").c_str());
        check_remove(("." + full_path + fname_base + "_B.lock").c_str());
        check_remove((full_path + fname_base + "_B.meta").c_str());
    }
}

visCalWriter::~visCalWriter() {
    restServer::instance().remove_get_callback(endpoint);
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
    json reply {"file_path", root_path + "/" + acq_name + "/" + fname_frozen};
    conn.send_json_reply(reply);
    INFO("Done. Resuming write loop.");
}

void visCalWriter::make_bundle(std::map<std::string, std::string>& metadata) {

    // Create the visFileBundle. This will not create any files until add_sample
    // is called
    file_bundle = std::unique_ptr<visCalFileBundle>(
        new visCalFileBundle(
            file_type, root_path, instrument_name, metadata, chunk_id, rollover,
            window, ds_id, file_length
        )
    );

    // TODO: is there a better way of using the child class method?
    file_cal_bundle = std::dynamic_pointer_cast<visCalFileBundle>(file_bundle);

    file_cal_bundle->set_file_name(fname_live, acq_name);
}
