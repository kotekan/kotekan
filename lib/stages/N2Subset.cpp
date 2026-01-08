#include "N2Subset.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for operator<
#include "N2FrameView.hpp"     // for N2FrameView, N2Field
#include "N2Util.hpp"          // for prod_ctype, frameID, cmap, icmap, modulo
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for FATAL_ERROR, WARN

#include "fmt.hpp"      // for compile_string_to_view
#include "gsl-lite.hpp" // for span

#include <algorithm>  // for max, copy, binary_search, sort
#include <complex>    // for complex
#include <functional> // for bind, function
#include <future>     // for future, async
#include <iterator>   // for back_insert_iterator, back_inserter
#include <set>        // for set
#include <stdint.h>   // for uint16_t, uint32_t
#include <utility>    // for pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Helper function: returns true if product contains at least one input from the list (OR condition)
static bool have_inputs_condition(const N2::prod_ctype& prod, const std::vector<int>& input_list) {
    for (int input : input_list) {
        if (prod.input_a == static_cast<uint16_t>(input)
            || prod.input_b == static_cast<uint16_t>(input)) {
            return true;
        }
    }
    return false;
}

// Helper function: returns true if product contains ONLY inputs from the list (AND condition)
static bool only_inputs_condition(const N2::prod_ctype& prod, const std::vector<int>& input_list) {
    bool a_in_list = false;
    bool b_in_list = false;
    for (int input : input_list) {
        if (prod.input_a == static_cast<uint16_t>(input))
            a_in_list = true;
        if (prod.input_b == static_cast<uint16_t>(input))
            b_in_list = true;
    }
    return a_in_list && b_in_list;
}

REGISTER_KOTEKAN_STAGE(N2Subset);

N2Subset::N2Subset(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2Subset::main_thread, this)),
    in_buf(get_buffer("in_buf")), out_buf(get_buffer("out_buf")),
    _prod_subset_list(parse_N2_subset(config, unique_name)),
    _num_subset_prod(std::get<0>(_prod_subset_list).size()),
    _base_prod_ind(std::get<0>(_prod_subset_list)),
    _base_prod_subset(std::get<1>(_prod_subset_list)) {

    // Get buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // in and out buffers must be of type N2Buffer
    if (in_buf->buffer_type != "N2" || out_buf->buffer_type != "N2") {
        FATAL_ERROR("N2Subset: both in_buf and out_buf must be of type 'N2'");
    }
}


void N2Subset::main_thread() {

    N2::frameID input_frame_id(in_buf);
    N2::frameID output_frame_id(out_buf);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (in_buf->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }

        N2FrameView input_vis(in_buf, input_frame_id);
        if (input_vis.n2_layout != N2Layout::FullUpperTri) {
            FATAL_ERROR("N2Subset: in_buf must have FullUpperTri visibility layout");
        }

        // Wait for the output buffer frame to be free
        if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
            break;
        }

        out_buf->allocate_new_metadata_object(output_frame_id);
        std::shared_ptr<N2Metadata> meta = get_N2_metadata(out_buf, output_frame_id);
        N2FrameView output_vis(out_buf, output_frame_id);

        // Make sure num_elements, ev matches
        if (input_vis.num_elements != output_vis.num_elements
            || input_vis.num_ev != output_vis.num_ev) {
            FATAL_ERROR("N2Subset: in_buf and out_buf must have the same num_elements and num_ev");
        }

        // Copy over metadata
        meta->deepCopy(input_vis._metadata);
        // Update metadata fields
        meta->num_prod = static_cast<uint32_t>(_num_subset_prod);
        // Set layout to GeneralSubset since we're extracting a subset of products
        meta->vis_layout = N2Layout::GeneralSubset;

        // Copy over subset of visibilities
        for (size_t i = 0; i < _num_subset_prod; i++) {
            output_vis.vis[i] = input_vis.vis[_base_prod_ind[i]];
            output_vis.weight[i] = input_vis.weight[_base_prod_ind[i]];
        }

        // Copy the non-visibility parts of the buffer
        output_vis.copy_data(input_vis, {N2Field::vis, N2Field::weight});
        // Mark the buffers and move on
        out_buf->mark_frame_full(unique_name, output_frame_id++);
        in_buf->mark_frame_empty(unique_name, input_frame_id++);
    }
}

std::tuple<std::vector<size_t>, std::vector<N2::prod_ctype>>
parse_N2_subset(Config& config, const std::string base_path) {

    size_t num_elements = config.get<size_t>(base_path, "num_elements");
    std::vector<size_t> prod_ind_vec;
    std::vector<N2::prod_ctype> prod_ctype_vec;

    // Type of product selection based on config parameter
    std::string prod_subset_type =
        config.get_default<std::string>(base_path, "prod_subset_type", "all");


    if (prod_subset_type == "Autocorrelations") {
        for (size_t i = 0; i < num_elements; i++) {
            prod_ind_vec.push_back(N2::cmap(i, i, num_elements));
            prod_ctype_vec.push_back({static_cast<uint16_t>(i), static_cast<uint16_t>(i)});
        }
    } else if (prod_subset_type == "have_inputs") {
        std::vector<int> input_list;
        input_list = config.get<std::vector<int>>(base_path, "input_list");
        // Find the products in the subset
        for (size_t ii = 0; ii < num_elements; ii++) {
            for (size_t jj = ii; jj < num_elements; jj++) {
                if (have_inputs_condition(
                        (N2::prod_ctype){static_cast<uint16_t>(ii), static_cast<uint16_t>(jj)},
                        input_list)) {
                    prod_ind_vec.push_back(N2::cmap(ii, jj, num_elements));
                    prod_ctype_vec.push_back(
                        {static_cast<uint16_t>(ii), static_cast<uint16_t>(jj)});
                }
            }
        }
    } else if (prod_subset_type == "only_inputs") {
        std::vector<int> input_list;
        input_list = config.get<std::vector<int>>(base_path, "input_list");
        // Find the products in the subset
        for (size_t ii = 0; ii < num_elements; ii++) {
            for (size_t jj = ii; jj < num_elements; jj++) {
                if (only_inputs_condition(
                        (N2::prod_ctype){static_cast<uint16_t>(ii), static_cast<uint16_t>(jj)},
                        input_list)) {
                    prod_ind_vec.push_back(N2::cmap(ii, jj, num_elements));
                    prod_ctype_vec.push_back(
                        {static_cast<uint16_t>(ii), static_cast<uint16_t>(jj)});
                }
            }
        }
    } else if (prod_subset_type == "all") {
        // Find the products in the subset
        for (size_t ii = 0; ii < num_elements; ii++) {
            for (size_t jj = ii; jj < num_elements; jj++) {
                prod_ind_vec.push_back(N2::cmap(ii, jj, num_elements));
                prod_ctype_vec.push_back({static_cast<uint16_t>(ii), static_cast<uint16_t>(jj)});
            }
        }
    }

    return std::make_tuple(prod_ind_vec, prod_ctype_vec);
}
