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

    // Check that the window isn't too long
    if (window > file_length) {
        std::string msg = fmt::format(
            "Active times window ({}) should not be greater than file length" \
            " ({}). Setting window to file length.", window, file_length
        );
        INFO(msg.c_str());
        window = file_length;
    }

    // TODO: get this from the datasetManager and put data type somewhere else
    instrument_name = config.get_default<std::string>(
                unique_name, "instrument_name", "chime");

    // Get the acq time out from the config
    acq_timeout = config.get_default<double>(unique_name, "acq_timeout", 300);
}

void visWriter::main_thread() {

    frameID frame_id(in_buf);

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // Check the dataset ID hasn't changed
        if (acqs.count(frame.dataset_id) == 0) {
            string msg = fmt::format(
                "Got new dataset ID={}. Starting a new acquisition.",
                frame.dataset_id
            );
            INFO(msg.c_str());

            // Try to initialise a new acquisition, if anything happens try to
            // exit gracefully
            try {
                init_acq(frame.dataset_id);
            }
            catch(const std::exception& e) {
                std::string msg = fmt::format(
                    "Error initialising acquisition output for dataset={}: {}",
                    frame.dataset_id, e.what()
                );
                ERROR(msg.c_str());
                raise(SIGINT);
                return;
            }
        }

        // Get the acquisition we are writing into
        auto& acq = acqs.at(frame.dataset_id);

        // Construct the new time
        auto ftime = frame.time;
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};

        // Check if the frequency we are receiving is on the list of frequencies
        // we are processing
        if (acq.freq_id_map.count(frame.freq_id) == 0) {
            WARN("Frequency id=%i not enabled for visWriter, discarding frame",
                 frame.freq_id);

        // Check that the number of visibilities matches what we expect
        } else if (frame.num_prod != acq.num_vis) {
            string msg = fmt::format(
                "Number of products in frame doesn't match state or file " \
                        "({} != {}).",
                frame.num_prod, acq.num_vis
            );
            ERROR(msg.c_str());
            raise(SIGINT);
            return;

        } else {

            DEBUG("Writing frequency id=%i", frame.freq_id);

            uint32_t freq_ind = acq.freq_id_map[frame.freq_id];

            // Add all the new information to the file.
            double start = current_time();
            write_mutex.lock();
            bool error = acq.file_bundle->add_sample(t, freq_ind, frame);
            write_mutex.unlock();
            double elapsed = current_time() - start;

            DEBUG("Write time %.5f s", elapsed);

            // Increase metric count if we dropped a frame at write time
            if(error) {
                // Relies on the fact that insertion zero intialises
                acq.dropped_frame_count[frame.freq_id] += 1;
                std::string labels = fmt::format("freq_id=\"{}\"," \
                                                 "dataset_id=\"{}\"",
                                                 frame.freq_id,
                                                 frame.dataset_id);
                prometheusMetrics::instance().add_process_metric(
                            "kotekan_viswriter_dropped_frame_total",
                            unique_name,
                            acq.dropped_frame_count.at(frame.freq_id),
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

        // Clean out any acquisitions that have been inactive long
        close_old_acqs();

    }
}


void visWriter::close_old_acqs()
{
    auto it = acqs.begin();
    while (it != acqs.end()) {
        auto last_update = it->second.file_bundle->last_update();
        double age = current_time() - last_update.ctime;

        if (age > acq_timeout) {
            it = acqs.erase(it);
        }
        else {
            it++;
        }
    }
}


void visWriter::get_dataset_state(dset_id_t ds_id) {

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

    // Get a reference to the acq state
    auto& acq = acqs.at(ds_id);

    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        acq.freq_id_map[f.first] = ind++;

    // compare git commit hashes
    // TODO: enforce and crash here if build type is Release?
    if (mstate->get_git_version_tag() != std::string(get_git_commit_hash())) {
        INFO("Git version tags don't match: dataset 0x%" PRIx64 " has tag %s," \
             "while the local git version tag is %s", ds_id,
             mstate->get_git_version_tag().c_str(),
             get_git_commit_hash());
    }

    acq.num_vis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();
}


void visWriter::init_acq(dset_id_t ds_id)
{
    // Create the new acqState
    auto& acq = acqs[ds_id];

    // get dataset states
    get_dataset_state(ds_id);

    // TODO: chunk ID is not really supported now. Just set it to zero.
    uint32_t chunk_id = 0;

    // Construct metadata
    auto metadata = make_metadata(ds_id);

    acq.file_bundle = std::make_unique<visFileBundle>(
        file_type, root_path, instrument_name, metadata, chunk_id, file_length,
        window, ds_id, file_length
    );
}


std::map<std::string, std::string> visWriter::make_metadata(dset_id_t ds_id)
{
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
    metadata["notes"] = "";   // TODO: connect up notes
    metadata["git_version_tag"] = get_git_commit_hash();
    metadata["system_user"] = user;
    metadata["collection_server"] = hostname;

    return metadata;
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


void visCalWriter::init_acq(dset_id_t ds_id)
{
    if (acqs.size() == 1) {
        throw std::runtime_error(
            "Cannot change dataset_id of visCalWriter. Exiting..."
        );
    }

    // Create the new acqState
    auto& acq = acqs[ds_id];

    // get dataset states
    get_dataset_state(ds_id);

    // TODO: chunk ID is not really supported now. Just set it to zero.
    uint32_t chunk_id = 0;

    // Construct metadata
    auto metadata = make_metadata(ds_id);

    // Create the derived class visCalFileBundle, and save to the instance, then
    // return as a unique_ptr
    file_cal_bundle = new visCalFileBundle(
        file_type, root_path, instrument_name, metadata, chunk_id, file_length,
        window, ds_id, file_length
    );
    file_cal_bundle->set_file_name(fname_live, acq_name);

    acq.file_bundle = std::unique_ptr<visFileBundle>(file_cal_bundle);
}

