#include "visWriter.hpp"
#include "visBuffer.hpp"
#include "util.h"
#include "errors.h"
#include "prodSubset.hpp"
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include "fpga_header_functions.h"
#include "prometheusMetrics.hpp"
#include "datasetManager.hpp"
#include "visCompression.hpp"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <time.h>
#include <regex>
#include "fmt.hpp"

REGISTER_KOTEKAN_PROCESS(visWriter);
REGISTER_KOTEKAN_PROCESS(visCalWriter);

visWriter::visWriter(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visWriter::main_thread, this)) {

    // Fetch any simple configuration
    root_path = config.get_string_default(unique_name, "root_path", ".");

    // Get the list of buffers that this process shoud connect to
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Get the type of the file we are writing
    // TODO: we may want to validate here rather than at creation time
    file_type = config.get_string_default(unique_name, "file_type", "hdf5fast");

    // If specified, get the weights type to write to attributes
    // TODO: add this to the datasetManager framework
    weights_type = config.get_string_default(unique_name, "weights_type", "unknown");

    file_length = config.get_int_default(unique_name, "file_length", 1024);
    window = config.get_int_default(unique_name, "window", 20);

    // Write the eigen values out? Communicated to visFile by num_ev > 0
    bool write_ev = config.get_bool_default(unique_name, "write_ev", false);
    num_ev = write_ev ? config.get_int(unique_name, "num_ev") : 0;

    use_dataset_manager = config.get_bool_default(
        unique_name, "use_dataset_manager", false);

    // If we are not using the dataset manager directly, create fake entries
    // from the config information supplied to the class
    if (!use_dataset_manager) {

        // Get the input labels and the products we are writing from the config
        auto ispec = std::get<1>(parse_reorder_default(config, unique_name));
        auto pspec = std::get<1>(parse_prod_subset(config, unique_name));

        // Get the frequency IDs we are going to write
        auto freq_id_list = config.get_array<uint32_t>(unique_name, "freq_ids");
        std::vector<std::pair<uint32_t, freq_ctype>> fspec;
        std::transform(freq_id_list.begin(), freq_id_list.end(),
                       std::back_inserter(fspec),
                       [](uint32_t f) -> std::pair<uint32_t, freq_ctype> {
                           return {f, {freq_from_bin(f), (400.0 / 1024)}};
                       });

        // Create the datasetState
        auto& dm = datasetManager::instance();

        // Construct a nested description of the initial state
        state_uptr freq_state = std::make_unique<freqState>(fspec);
        state_uptr input_state = std::make_unique<inputState>(
            ispec, std::move(freq_state));
        state_uptr prod_state = std::make_unique<prodState>(
            pspec, std::move(input_state));

        // Register the initial state with the manager
        auto s = dm.add_state(std::move(prod_state));
        writer_dstate = s.first;
    }

    // TODO: long term this should come from some dynamic source (dM?)
    instrument_name = config.get_string_default(unique_name, "instrument_name", "chime");

    // Set the instrument name from the hostname if in node mode
    node_mode = config.get_bool_default(unique_name, "node_mode", false);
    if(node_mode) {
        std::string t(256, '\0');
        gethostname(&t[0], 256);
        // Here we trim the hostname to the first alphanumeric segment only.
        instrument_name = t.substr(0, (t + ".").find_first_of(".-"));
    }
}

void visWriter::apply_config(uint64_t fpga_seq) {

}

void visWriter::main_thread() {

    unsigned int frame_id = 0;

    // Look over the current buffers for information to setup the acquisition
    if (!init_acq())
        return;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if(wait_for_full_frame(in_buf, unique_name.c_str(),
                               frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // Construct the new time
        auto ftime = frame.time;
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};

        // Check if the frequency we are receiving is on the list of frequencies
        // we are processing
        if (freq_id_map.count(frame.freq_id) == 0) {
            WARN("Frequency id=%i not enabled for visWriter, discarding frame",
                 frame.freq_id);

        // Check that the number of visibilities matches what we expect
        } else if (frame.num_prod != num_vis) {
            string msg = fmt::format(
                "Number of products in frame doesn't match file ({} != {}).",
                frame.num_prod, num_vis
            );
            ERROR(msg.c_str());
            raise(SIGINT);
            return;

        // Check the number of eigen vectors is as expected
        } else if (num_ev > 0  and frame.num_ev != num_ev) {

            string msg = fmt::format(
                "Number of eigenvectors in frame doesn't match file ({} != {}).",
                frame.num_ev, num_ev
            );
            ERROR(msg.c_str());
            raise(SIGINT);
            return;

        // Check the dataset ID hasn't changed
        } else if (use_dataset_manager && frame.dataset_id != dataset) {
            string msg = fmt::format(
                "Unexpected dataset ID={} received (expected id={}).",
                frame.dataset_id, dataset
            );
            ERROR(msg.c_str());
            raise(SIGINT);
            return;

        } else {

            INFO("Writing frequency id=%i", frame.freq_id);

            uint32_t freq_ind = freq_id_map[frame.freq_id];

            // Add all the new information to the file.
            double start = current_time();
            write_mutex.lock();
            bool error = file_bundle->add_sample(t, freq_ind, frame);
            write_mutex.unlock();
            double elapsed = current_time() - start;

            DEBUG("Write time %.5f s", elapsed);

            // Increase metric count if we dropped a frame at write time
            if(error) {
                dropped_frame_count++;
                prometheusMetrics::instance().add_process_metric(
                    "kotekan_viswriter_dropped_frame_total",
                    unique_name, dropped_frame_count
                );
            }

            // Update average write time in prometheus
            write_time.add_sample(elapsed);
            prometheusMetrics::instance().add_process_metric(
                "kotekan_viswriter_write_time_seconds",
                unique_name, write_time.average()
            );

        }

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);

        // Advance the current frame ids
        frame_id = (frame_id + 1) % in_buf->num_frames;

    }
}



bool visWriter::init_acq() {

    // Fetch out information from the buffers that are needed for setting  up
    // the acq. For the moment just read the first frame.
    if (wait_for_full_frame(in_buf, unique_name.c_str(), 0) == nullptr)
        return false;
    auto frame = visFrameView(in_buf, 0);

    auto& dm = datasetManager::instance();

    // Get the dataset ID that the downstream files can use to determine what
    // they will be writing.
    if (use_dataset_manager) {
        dataset = frame.dataset_id;
    } else {
        dataset = dm.add_dataset(writer_dstate, -1);
    }

    // Get the frequency spec to determine the freq_ids expected at this Writer.
    auto fstate = dm.closest_ancestor_of_type<freqState>(
        frame.dataset_id).second;
    uint ind = 0;
    for (auto& f : fstate->get_freqs())
        freq_id_map[f.first] = ind++;

    // Get the product spec and (if available) the stackState to determine the
    // number of vis entries we are expecting
    auto pstate = dm.closest_ancestor_of_type<prodState>(dataset).second;
    auto sstate = dm.closest_ancestor_of_type<stackState>(dataset).second;
    num_vis = sstate ? sstate->get_num_stack() : pstate->get_prods().size();

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
    metadata["git_version_tag"] = "not set";
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

    return true;
}


void visWriter::make_bundle(std::map<std::string, std::string>& metadata) {
    file_bundle = std::make_unique<visFileBundle>(
        file_type, root_path, instrument_name, metadata, chunk_id,
        rollover, window, dataset, num_ev, file_length
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
    std::string fname_base = config.get_string_default(unique_name, "file_base", "cal");
    acq_name = config.get_string_default(unique_name, "dir_name", "cal");
    // Initially start with this buffer configuration
    fname_live = fname_base + "_A";
    fname_frozen = fname_base + "_B";

    // Use a very short window by default
    window = config.get_int_default(unique_name, "window", 10);

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
            window, dataset, num_ev, file_length
        )
    );

    // TODO: is there a better way of using the child class method?
    file_cal_bundle = std::dynamic_pointer_cast<visCalFileBundle>(file_bundle);

    file_cal_bundle->set_file_name(fname_live, acq_name);
}
