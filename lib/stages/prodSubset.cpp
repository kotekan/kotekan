#include "prodSubset.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for operator<
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for allocate_new_metadata_object, mark_frame_empty, mark_fram...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, state_id_t, datasetManager
#include "datasetState.hpp"    // for prodState
#include "kotekanLogging.hpp"  // for FATAL_ERROR, WARN
#include "visBuffer.hpp"       // for VisFrameView, VisField, VisField::vis, VisField::weight
#include "visUtil.hpp"         // for prod_ctype, frameID, cmap, icmap, modulo, cfloat

#include "gsl-lite.hpp" // for span

#include <algorithm>    // for max, binary_search, copy, sort
#include <atomic>       // for atomic_bool
#include <complex>      // for complex
#include <cxxabi.h>     // for __forced_unwind
#include <exception>    // for exception
#include <functional>   // for _Bind_helper<>::type, bind, function
#include <future>       // for future, async
#include <iterator>     // for back_insert_iterator, back_inserter
#include <regex>        // for match_results<>::_Base_type
#include <stdexcept>    // for out_of_range, runtime_error
#include <stdint.h>     // for uint16_t, uint32_t
#include <system_error> // for system_error
#include <utility>      // for pair, tuple_element<>::type


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(prodSubset);

prodSubset::prodSubset(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&prodSubset::main_thread, this)) {

    // Get buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    // TODO: Size of buffer is not adjusted for baseline subset.
    //       ~ 3/4 of buffer will be unused.
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    auto subset_list = parse_prod_subset(config, unique_name);
    _base_prod_ind = std::get<0>(subset_list);
    _base_prod_subset = std::get<1>(subset_list);
}

void prodSubset::change_dataset_state(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();

    auto fprint = dm.fingerprint(ds_id, {"products"});

    if (states_map.count(fprint) == 0) {

        // create new product dataset state
        const prodState* prod_state_ptr = dm.dataset_state<prodState>(ds_id);
        if (prod_state_ptr == nullptr) {
            FATAL_ERROR(
                "Set to not use dataset_broker and couldn't find prodState ancestor of dataset "
                "{}. Make sure there is a stage upstream in the config, that adds a "
                "prodState.\nExiting...",
                ds_id);
        }

        // Compare function to help sort input_ctypes.
        auto compare_prods = [](const prod_ctype& a, const prod_ctype& b) {
            return (a.input_a < b.input_a) || (a.input_a == b.input_a && a.input_b < b.input_b);
        };

        // get a sorted copy of input prods
        const std::vector<prod_ctype>& input_prods = prod_state_ptr->get_prods();
        std::vector<prod_ctype> input_prods_copy;
        std::copy(input_prods.begin(), input_prods.end(), std::back_inserter(input_prods_copy));
        std::sort(input_prods_copy.begin(), input_prods_copy.end(), compare_prods);

        std::vector<size_t> new_prod_ind;
        std::vector<prod_ctype> new_prod_subset;

        // check if prod_subset is a subset of the prodState
        for (size_t i = 0; i < _base_prod_subset.size(); i++) {
            // so we can use binary search
            if (std::binary_search(input_prods_copy.begin(), input_prods_copy.end(),
                                   _base_prod_subset.at(i), compare_prods)) {
                new_prod_ind.push_back(_base_prod_ind[i]);
                new_prod_subset.push_back(_base_prod_subset[i]);
            } else {
                WARN("prodSubset: Product ID {:d} is configured to be in the subset, but is "
                     "missing in "
                     "dataset {}. Deleting it from subset.",
                     _base_prod_ind.at(i), ds_id);
            }
        }

        auto state_id = dm.create_state<prodState>(new_prod_subset).first;

        states_map[fprint] = {state_id, new_prod_ind};
    }

    auto [state_id, prod_ind] = states_map.at(fprint);
    dset_id_map[ds_id] = {dm.add_dataset(state_id, ds_id), prod_ind};
}

void prodSubset::main_thread() {


    frameID input_frame_id(in_buf);
    frameID output_frame_id(out_buf);

    std::future<void> change_dset_fut;

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (in_buf->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = VisFrameView(in_buf, input_frame_id);

        // check if the input dataset has changed
        if (dset_id_map.count(input_frame.dataset_id) == 0) {
            change_dset_fut =
                std::async(&prodSubset::change_dataset_state, this, input_frame.dataset_id);
        }

        // Wait for the output buffer frame to be free
        if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
            break;
        }

        // Allocate metadata and get output frame
        out_buf->allocate_new_metadata_object(output_frame_id);

        if (change_dset_fut.valid())
            change_dset_fut.wait();

        auto& [new_dset_id, prod_ind] = dset_id_map.at(input_frame.dataset_id);

        size_t subset_num_prod = prod_ind.size();

        // Create view to output frame
        auto output_frame =
            VisFrameView::create_frame_view(out_buf, output_frame_id, input_frame.num_elements,
                                            subset_num_prod, input_frame.num_ev);

        // Copy over subset of visibilities
        for (size_t i = 0; i < subset_num_prod; i++) {
            output_frame.vis[i] = input_frame.vis[prod_ind[i]];
            output_frame.weight[i] = input_frame.weight[prod_ind[i]];
        }

        // Copy metadata
        output_frame.copy_metadata(input_frame);
        output_frame.dataset_id = new_dset_id;

        // Copy the non-visibility parts of the buffer
        output_frame.copy_data(input_frame, {VisField::vis, VisField::weight});

        // Mark the buffers and move on
        out_buf->mark_frame_full(unique_name, output_frame_id++);
        in_buf->mark_frame_empty(unique_name, input_frame_id++);
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
