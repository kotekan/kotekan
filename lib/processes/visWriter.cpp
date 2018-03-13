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
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <time.h>
#include "fmt.hpp"

REGISTER_KOTEKAN_PROCESS(visWriter);

visWriter::visWriter(Config& config,
                       const string& unique_name,
                       bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&visWriter::main_thread, this)) {

    // Fetch any simple configuration
    // num_freq = config.get_int(unique_name, "num_freq");
    root_path = config.get_string_default(unique_name, "root_path", ".");

    // Get the list of buffers that this process shoud connect to
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Get the input labels
    inputs = std::get<1>(parse_reorder_default(config, unique_name));

    // If specified, get the weights type to write to attributes
    weights_type = config.get_string_default(unique_name, "weights_type", "unknown");

    // Write the eigen values out? Communicated to visFile by num_ev > 0
    bool write_ev = config.get_bool_default(unique_name, "write_ev", false);
    num_ev = write_ev ? config.get_int(unique_name, "num_ev") : 0;

    // TODO: dynamic setting of instrument name, shouldn't be hardcoded here, At
    // the moment this either uses chime, or if set to use a per_node_instrument
    // it uses the hostname of the current node
    node_mode = config.get_bool_default(unique_name, "node_mode", true);

    // Type of product selection based on config parameter
    prod_subset_type = config.get_string_default(unique_name, "prod_subset_type", "all");

    if (prod_subset_type == "autos") {
    // If this writter is auto-correlations only:
        for (uint16_t ii=0; ii < inputs.size(); ii++) {
            prods.push_back({ii,ii});
        }
    } else if (prod_subset_type == "baseline") {
        // If this writer takes a baseline subset as input, generate 
        // subset products as specified in config.
        xmax = config.get_int(unique_name, "max_ew_baseline");
        ymax = config.get_int(unique_name, "max_ns_baseline");
        for(uint16_t ii=0; ii < inputs.size(); ii++) {
            for(uint16_t jj = ii; jj < inputs.size(); jj++) {
                // restrict products to those within max baseline length
                if (max_bl_condition({ii, jj}, xmax, ymax)) {
                    prods.push_back({ii, jj});
                }
            }
        }
    } else if (prod_subset_type == "input_list") {
        input_list = config.get_int_array(unique_name, "input_list");
        // Find the products in the subset
        for(uint16_t ii=0; ii < inputs.size(); ii++) {
            for(uint16_t jj = ii; jj < inputs.size(); jj++) {
                // restrict products to those containing selected inputs
                if (input_list_condition({ii,jj}, input_list)) {
                    prods.push_back({ii, jj});
                }
            }
        }
    } else if (prod_subset_type == "all") {
        // Select all rpoducts
        for(uint16_t ii=0; ii < inputs.size(); ii++) {
            for(uint16_t jj = ii; jj < inputs.size(); jj++) {
                prods.push_back({ii, jj});
            }
        }
    }

    num_prod = prods.size();

    if(node_mode) {

        // Set the instrument_name from the hostname
        char temp[256];
        gethostname(temp, 256);
        std::string t = temp;
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

            // Create fake entries to fill out the gain and weight datasets with
            // because these don't correctly make it through kotekan yet
            // TODO: these should be read directly from the span
            std::vector<cfloat> vis(frame.vis.begin(), frame.vis.end());
            std::vector<float> vis_weight(frame.weight.begin(), frame.weight.end());
            std::vector<cfloat> gain_coeff(inputs.size(), {1, 0});
            std::vector<int32_t> gain_exp(inputs.size(), 0);
            std::vector<float> eval(frame.eval.begin(), frame.eval.end());
            std::vector<cfloat> evec(frame.evec.begin(), frame.evec.end());

            // Add all the new information to the file.
            double start = current_time();
            bool error = file_bundle->addSample(t, freq_ind, vis, vis_weight,
                                                gain_coeff, gain_exp, eval,
                                                evec, frame.erms);
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
    std::vector<uint32_t> freq_ids;

    wait_for_full_frame(in_buf, unique_name.c_str(), frame_id);

    auto frame = visFrameView(in_buf, frame_id);
    freq_ids.push_back(frame.freq_id);

    // Use the per buffer info to setup the acqusition properties
    setup_freq(freq_ids);

    // Create the visFileBundle. This will not create any files until addSample
    // is called
    // TODO: connect up notes
    std::string notes = "";
    file_bundle = std::unique_ptr<visFileBundle>(
        new visFileBundle(
            root_path, instrument_name, chunk_id, file_length, window,
            notes, weights_type, freqs, inputs, prods, num_ev
        )
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
            char t[32];
            snprintf(t, 32, "%i [%.2f MHz] ", id, freq_from_bin(id));
            s += t;
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
