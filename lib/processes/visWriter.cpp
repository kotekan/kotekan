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

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <time.h>
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

    // Get the input labels
    inputs = std::get<1>(parse_reorder_default(config, unique_name));

    // If specified, get the weights type to write to attributes
    weights_type = config.get_string_default(unique_name, "weights_type", "unknown");

    file_length = config.get_int_default(unique_name, "file_length", 1024);
    window = config.get_int_default(unique_name, "window", 20);

    // Write the eigen values out? Communicated to visFile by num_ev > 0
    bool write_ev = config.get_bool_default(unique_name, "write_ev", false);
    num_ev = write_ev ? config.get_int(unique_name, "num_ev") : 0;

    node_mode = config.get_bool_default(unique_name, "node_mode", false);

    // Calculate the set of products we are writing from the config
    prods = std::get<1>(parse_prod_subset(config, unique_name));
    num_prod = prods.size();

    // TODO: dynamic setting of instrument name, shouldn't be hardcoded here, At
    // the moment this either uses chime, or if set to use a per_node_instrument
    // it uses the hostname of the current node
    if(node_mode) {

        // Set the instrument_name from the hostname
        std::string t(256, '\0');
        gethostname(&t[0], 256);
        // Here we trim the hostname to the first alphanumeric segment only.
        instrument_name = t.substr(0, (t + ".").find_first_of(".-"));

        node_mode = true;
        num_freq = 4;

    } else {
        instrument_name = config.get_string_default(unique_name, "instrument_name", "chime");
        freq_id_list = config.get_int_array(unique_name, "freq_ids");
    }
}

void visWriter::apply_config(uint64_t fpga_seq) {

}

void visWriter::main_thread() {

    unsigned int frame_id = 0;

    // Look over the current buffers for information to setup the acquisition
    init_acq();

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

        // Check if the frequency we are receiving is on the list of frequencies we are processing
        if(freq_map.count(frame.freq_id) == 0) {
            WARN("Frequency id=%i is not enabled for visWriter, discarding frame", frame.freq_id);
        } else if (frame.num_prod != num_prod) {

            string msg = fmt::format(
                "Number of products in frame doesn't match file ({} != {}).", frame.num_prod, num_prod
            );
            throw std::runtime_error(msg);

        } else if (num_ev > 0  and frame.num_ev != num_ev) {

            string msg = fmt::format(
                "Number of eigenvectors in frame doesn't match file ({} != {}).",
                frame.num_ev, num_ev
            );
            throw std::runtime_error(msg);

        } else {

            INFO("Writing frequency id=%i", frame.freq_id);

            // Lookup the frequency index if reordering, otherwise write out in buffer order
            uint32_t freq_ind = freq_map[frame.freq_id];

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



void visWriter::init_acq() {

    // Fetch out information from the buffers that are needed for setting  up
    // the acq. For the moment just read the first frame.
    unsigned int frame_id = 0;

    wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);

    auto frame = visFrameView(in_buf, frame_id);

    if (use_dataset_manager) {
        auto& dm = datasetManager::instance();

        auto f = dm.closest_ancestor_of_type<freqState>(frame.dataset_id);
        if (f.second == nullptr) {
            throw std::runtime_error(
                "Required freqState not available for dataset_id=%i",
                frame.dataset_id
            )
        }
        for(auto& f : f.second->get_freqs();

        if (f.second == nullptr) {
            throw std::runtime_error(
                "Required freqState not available for dataset_id=%i",
                frame.dataset_id
            )
        }
    }
    else {
        // Use the per buffer info to setup the acqusition properties
        setup_freq(frame.freq_id);
    }

    // Get the current user
    std::string user(256, '\0');
    user = (getlogin_r(&user[0], 256) == 0) ? user.c_str() : "unknown";

    // Get the current hostname of the system
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
}


void visWriter::make_bundle(std::map<std::string, std::string>& metadata) {

    // Create the visFileBundle. This will not create any files until add_sample
    // is called
    file_bundle = std::make_unique<visFileBundle>(
        file_type, root_path, instrument_name, metadata, chunk_id, rollover,
        window, freqs, inputs, prods, num_ev, file_length
    );
}


void visWriter::setup_freq(const std::vector<uint32_t>& freq_ids) {
    // TODO: this function needs to do three things: set the frequency input map
    // (freqs), set the frequency ordering (freq_map) and set the chunk ID
    // (chunk_id).
    if(node_mode) {
        // Output all the frequencies that we have found
        std::string s;
        for(auto id : freq_ids) {
            s += fmt::format("{} [{:.2f} MHz] ", id, freq_from_bin(id));
        }
        INFO("Frequency bins found: %s", s.c_str());

        // TODO: this uses the hacky way of deriving the chunk ID
        unsigned int node_id = freq_ids[0] % 256;

        // Set the list of frequency ids that this node will deal with
        for(unsigned int i = 0; i < 4; i++) {
            freq_id_list.push_back(256 * i + node_id);
        }
    }

    // Set the chunk_id
    // TODO: eventually this should be set properly, but at the moment as we are
    // not merging into a single dir, chunk_id=0 should always be fine
    chunk_id = 0;

    // Sort the streams into bin order, this will give the order in which they
    // are written out
    unsigned int fpos = 0;
    for(auto id : freq_id_list) {
        freq_map[id] = fpos;
        freqs.push_back({freq_from_bin(id), (400.0 / 1024)});
        fpos++;
    }
}


visCalWriter::visCalWriter(Config &config,
                            const string& unique_name,
                            bufferContainer &buffer_container) :
    visWriter::visWriter(config, unique_name, buffer_container) {

    // Register REST callback
    endpoint = "/release_cal_file";
    using namespace std::placeholders;
    restServer::instance().register_get_callback(endpoint,
            std::bind(&visCalWriter::rest_callback, this, _1));

    // Get file name to write to
    // TODO: strip file extensions?
    file_name = config.get_string_default(unique_name, "file_name", "live");
    acq_name = config.get_string_default(unique_name, "dir_name", "cal");
    frozen_file_name = config.get_string_default(unique_name,
                                                 "frozen_file_name", "cal");

    // Force use of VisFileRing
    file_type = "ring";

    // Check if any of these files exist
    std::string full_path = root_path + "/" + acq_name + "/";
    if (access((full_path + file_name + ".data").c_str(), F_OK) == 0) {
        // Delete existing files
        INFO(("Clobering files " + full_path + file_name + ".*").c_str());
        check_remove(full_path + file_name + ".data");
        check_remove(full_path + file_name + ".meta");
    }
}

visCalWriter::~visCalWriter() {
    restServer::instance().remove_get_callback(endpoint);
}

void visCalWriter::rest_callback(connectionInstance& conn) {
    // Ensure no write is ongoing
    std::lock_guard<std::mutex> write_guard(write_mutex);

    INFO("Received request to release calibration live file...");

    // Tell visCalFileBundle to stop writing to current file
    file_cal_bundle->clear_file_map();
    // Remove previous frozen buffer and replace with new one
    std::string full_path = root_path + "/" + acq_name;
    INFO(("Updating cal file names in " + full_path).c_str());
    check_remove(full_path + "/" + frozen_file_name + ".data");
    check_remove(full_path + "/" + frozen_file_name + ".meta");
    check_rename(full_path + "/" + file_name + ".data",
                 full_path + "/" + frozen_file_name + ".data");
    check_rename(full_path + "/" + file_name + ".meta",
                 full_path + "/" + frozen_file_name + ".meta");

    // Respond with frozen file path
    json reply {
        {"file_path", root_path + "/" + acq_name + "/" + frozen_file_name}
    };
    conn.send_json_reply(reply);
    INFO("Done. Resuming write loop.");
}

void visCalWriter::make_bundle(std::map<std::string, std::string>& metadata) {

    // Create the visFileBundle. This will not create any files until add_sample
    // is called
    file_bundle = std::unique_ptr<visCalFileBundle>(
        new visCalFileBundle(
            file_type, root_path, instrument_name, metadata, chunk_id, rollover,
            window, freqs, inputs, prods, num_ev, file_length
        )
    );

    // TODO: is there a better way of using the child class method?
    file_cal_bundle = std::dynamic_pointer_cast<visCalFileBundle>(file_bundle);

    file_cal_bundle->set_file_name(file_name, acq_name);
}
