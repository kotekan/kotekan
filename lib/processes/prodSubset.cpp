#include "prodSubset.hpp"

#include "datasetManager.hpp"
#include "datasetState.hpp"
#include "errors.h"
#include "processFactory.hpp"
#include "prometheusMetrics.hpp"
#include "visBuffer.hpp"
#include "visUtil.hpp"

#include "gsl-lite.hpp"

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cxxabi.h>
#include <exception>
#include <functional>
#include <inttypes.h>
#include <iterator>
#include <memory>
#include <regex>
#include <signal.h>
#include <stdexcept>
#include <utility>


REGISTER_KOTEKAN_PROCESS(prodSubset);

prodSubset::prodSubset(Config& config, const string& unique_name,
                       bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&prodSubset::main_thread, this)) {

    // Get buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    // TODO: Size of buffer is not adjusted for baseline subset.
    //       ~ 3/4 of buffer will be unused.
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    auto subset_list = parse_prod_subset(config, unique_name);
    prod_ind = std::get<0>(subset_list);
    prod_subset = std::get<1>(subset_list);

    subset_num_prod = prod_ind.size();
}

dset_id_t prodSubset::change_dataset_state(dset_id_t ds_id, std::vector<prod_ctype>& prod_subset,
                                           std::vector<size_t>& prod_ind, size_t& subset_num_prod) {
    auto& dm = datasetManager::instance();

    // create new product dataset state
    const prodState* prod_state_ptr = dm.dataset_state<prodState>(ds_id);
    if (prod_state_ptr == nullptr) {
        ERROR("Set to not use dataset_broker and couldn't find "
              "freqState ancestor of dataset 0x%" PRIx64 ". Make sure there "
              "is a process upstream in the config, that adds a freqState.\n"
              "Exiting...",
              ds_id);
        raise(SIGINT);
    }

    // get a copy of input prods
    const std::vector<prod_ctype>& input_prods = prod_state_ptr->get_prods();
    std::vector<prod_ctype> input_prods_copy;
    std::copy(input_prods.begin(), input_prods.end(), std::back_inserter(input_prods_copy));

    // Compare function to help sort input_ctypes.
    auto compare_prods = [](const prod_ctype& a, const prod_ctype& b) {
        return (a.input_a < b.input_a) || (a.input_a == b.input_a && a.input_b < b.input_b);
    };

    // sort them for binary search
    std::sort(input_prods_copy.begin(), input_prods_copy.end(), compare_prods);

    // check if prod_subset is a subset of the prodState
    for (size_t i = 0; i < prod_subset.size(); i++) {
        // so we can use binary search
        if (!std::binary_search(input_prods_copy.begin(), input_prods_copy.end(), prod_subset.at(i),
                                compare_prods)) {
            WARN("prodSubset: Product ID 0x%" PRIx64 " is configured to be in"
                 "the subset, but is missing in dataset 0x%" PRIx64 ". "
                 "Deleting it from subset.",
                 prod_ind.at(i), ds_id);
            prod_subset.erase(prod_subset.cbegin() + i);
            prod_ind.erase(prod_ind.cbegin() + i);
        }
    }

    // in case we deleted sth, save new size
    subset_num_prod = prod_subset.size();

    state_uptr pstate = std::make_unique<prodState>(prod_subset);
    state_id_t prod_state_id = dm.add_state(std::move(pstate)).first;
    return dm.add_dataset(ds_id, prod_state_id);
}

void prodSubset::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;

    dset_id_t input_dset_id;
    dset_id_t output_dset_id = 0;

    // Wait for a frame in the input buffer in order to get the dataset ID
    if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
        return;
    }
    input_dset_id = visFrameView(in_buf, input_frame_id).dataset_id;
    future_output_dset_id =
        std::async(&prodSubset::change_dataset_state, this, input_dset_id, std::ref(prod_subset),
                   std::ref(prod_ind), std::ref(subset_num_prod));

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // check if the input dataset has changed
        if (input_dset_id != input_frame.dataset_id) {
            input_dset_id = input_frame.dataset_id;
            future_output_dset_id =
                std::async(&prodSubset::change_dataset_state, this, input_dset_id,
                           std::ref(prod_subset), std::ref(prod_ind), std::ref(subset_num_prod));
        }

        // Wait for the output buffer frame to be free
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }

        // Allocate metadata and get output frame
        allocate_new_metadata_object(out_buf, output_frame_id);

        // Are we waiting for a new dataset ID?
        if (future_output_dset_id.valid())
            output_dset_id = future_output_dset_id.get();

        // Create view to output frame
        auto output_frame = visFrameView(out_buf, output_frame_id, input_frame.num_elements,
                                         subset_num_prod, input_frame.num_ev);

        // Copy over subset of visibilities
        for (size_t i = 0; i < subset_num_prod; i++) {
            output_frame.vis[i] = input_frame.vis[prod_ind[i]];
            output_frame.weight[i] = input_frame.weight[prod_ind[i]];
        }

        // Copy metadata
        output_frame.copy_metadata(input_frame);
        // Copy the non-visibility parts of the buffer
        output_frame.copy_data(input_frame, {visField::vis, visField::weight});

        // set the dataset ID in the outgoing frame
        output_frame.dataset_id = output_dset_id;

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);

        // Advance the current frame id
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
    }
}

inline bool max_bl_condition(prod_ctype prod, int xmax, int ymax) {

    // Figure out feed separations
    int x_sep = prod.input_a / 512 - prod.input_b / 512;
    int y_sep = prod.input_a % 256 - prod.input_b % 256;
    if (x_sep < 0)
        x_sep = -x_sep;
    if (y_sep < 0)
        y_sep = -y_sep;

    return (x_sep <= xmax) && (y_sep <= ymax);
}

inline bool max_bl_condition(uint32_t vis_ind, int n, int xmax, int ymax) {

    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return max_bl_condition(prod, xmax, ymax);
}

inline bool have_inputs_condition(prod_ctype prod, std::vector<int> input_list) {

    bool prod_in_list = false;
    for (auto ipt : input_list) {
        if ((prod.input_a == ipt) || (prod.input_b == ipt)) {
            prod_in_list = true;
            break;
        }
    }

    return prod_in_list;
}

inline bool have_inputs_condition(uint32_t vis_ind, int n, std::vector<int> input_list) {

    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return have_inputs_condition(prod, input_list);
}

inline bool only_inputs_condition(prod_ctype prod, std::vector<int> input_list) {

    bool ipta_in_list = false;
    bool iptb_in_list = false;
    for (auto ipt : input_list) {
        if (prod.input_a == ipt) {
            ipta_in_list = true;
        }
        if (prod.input_b == ipt) {
            iptb_in_list = true;
        }
    }

    return (ipta_in_list && iptb_in_list);
}

inline bool only_inputs_condition(uint32_t vis_ind, int n, std::vector<int> input_list) {

    // Get product indices
    prod_ctype prod = icmap(vis_ind, n);

    return only_inputs_condition(prod, input_list);
}


std::tuple<std::vector<size_t>, std::vector<prod_ctype>>
parse_prod_subset(Config& config, const std::string base_path) {

    size_t num_elements = config.get<size_t>(base_path, "num_elements");
    std::vector<size_t> prod_ind_vec;
    std::vector<prod_ctype> prod_ctype_vec;

    // Type of product selection based on config parameter
    std::string prod_subset_type =
        config.get_default<std::string>(base_path, "prod_subset_type", "all");


    if (prod_subset_type == "autos") {
        for (uint16_t ii = 0; ii < num_elements; ii++) {
            prod_ind_vec.push_back(cmap(ii, ii, num_elements));
            prod_ctype_vec.push_back({ii, ii});
            //            prod_ctype_vec.emplace_back((prod_ctype){ii,ii});
        }
    } else if (prod_subset_type == "baseline") {
        // Define criteria for baseline selection based on config parameters
        uint16_t xmax, ymax;
        xmax = config.get<uint16_t>(base_path, "max_ew_baseline");
        ymax = config.get<uint16_t>(base_path, "max_ns_baseline");
        // Find the products in the subset
        for (uint16_t ii = 0; ii < num_elements; ii++) {
            for (uint16_t jj = ii; jj < num_elements; jj++) {
                if (max_bl_condition((prod_ctype){ii, jj}, xmax, ymax)) {
                    prod_ind_vec.push_back(cmap(ii, jj, num_elements));
                    prod_ctype_vec.push_back({ii, jj});
                    //                    prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
                }
            }
        }
    } else if (prod_subset_type == "have_inputs") {
        std::vector<int> input_list;
        input_list = config.get<std::vector<int>>(base_path, "input_list");
        // Find the products in the subset
        for (uint16_t ii = 0; ii < num_elements; ii++) {
            for (uint16_t jj = ii; jj < num_elements; jj++) {
                if (have_inputs_condition((prod_ctype){ii, jj}, input_list)) {
                    prod_ind_vec.push_back(cmap(ii, jj, num_elements));
                    prod_ctype_vec.push_back({ii, jj});
                    //                    prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
                }
            }
        }
    } else if (prod_subset_type == "only_inputs") {
        std::vector<int> input_list;
        input_list = config.get<std::vector<int>>(base_path, "input_list");
        // Find the products in the subset
        for (uint16_t ii = 0; ii < num_elements; ii++) {
            for (uint16_t jj = ii; jj < num_elements; jj++) {
                if (only_inputs_condition((prod_ctype){ii, jj}, input_list)) {
                    prod_ind_vec.push_back(cmap(ii, jj, num_elements));
                    prod_ctype_vec.push_back({ii, jj});
                    //                    prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
                }
            }
        }
    } else if (prod_subset_type == "all") {
        // Find the products in the subset
        for (uint16_t ii = 0; ii < num_elements; ii++) {
            for (uint16_t jj = ii; jj < num_elements; jj++) {
                prod_ind_vec.push_back(cmap(ii, jj, num_elements));
                prod_ctype_vec.push_back({ii, jj});
                //              prod_ctype_vec.emplace_back((prod_ctype){ii,jj});
            }
        }
    }

    return std::make_tuple(prod_ind_vec, prod_ctype_vec);
}
