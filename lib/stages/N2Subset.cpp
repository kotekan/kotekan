#include "N2Subset.hpp"

#include "Config.hpp"          // for Config
#include "N2FrameDesc.hpp"     // for N2FrameDesc
#include "N2FrameView.hpp"     // for N2FrameView, N2Field
#include "N2Util.hpp"          // for prod_ctype, frameID
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO

#include "fmt.hpp" // for format

#include <map>     // for map
#include <utility> // for pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::N2Field;
using kotekan::N2FrameDesc;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(N2Subset);

N2Subset::N2Subset(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2Subset::main_thread, this)) {

    // Get buffers
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Validate buffer types
    if (in_buf->buffer_type != "N2") {
        FATAL_ERROR("N2Subset: in_buf must be of type 'N2', got '{:s}'", in_buf->buffer_type);
    }
    if (out_buf->buffer_type != "N2") {
        FATAL_ERROR("N2Subset: out_buf must be of type 'N2', got '{:s}'", out_buf->buffer_type);
    }

    // Get and validate frame descriptors
    auto in_frame_desc = in_buf->get_frame_description();
    if (!in_frame_desc) {
        FATAL_ERROR("N2Subset: in_buf does not have a frame descriptor set");
    }
    in_desc = std::dynamic_pointer_cast<const N2FrameDesc>(in_frame_desc);
    if (!in_desc) {
        FATAL_ERROR("N2Subset: in_buf does not have an N2FrameDesc");
    }

    auto out_frame_desc = out_buf->get_frame_description();
    if (!out_frame_desc) {
        FATAL_ERROR("N2Subset: out_buf does not have a frame descriptor set");
    }
    out_desc = std::dynamic_pointer_cast<const N2FrameDesc>(out_frame_desc);
    if (!out_desc) {
        FATAL_ERROR("N2Subset: out_buf does not have an N2FrameDesc");
    }

    // Validate num_elements and num_ev match (required for copying non-product fields)
    if (in_desc->get_num_elements() != out_desc->get_num_elements()) {
        FATAL_ERROR("N2Subset: num_elements mismatch: in_buf has {:d}, out_buf has {:d}",
                    in_desc->get_num_elements(), out_desc->get_num_elements());
    }
    if (in_desc->get_num_ev() != out_desc->get_num_ev()) {
        FATAL_ERROR("N2Subset: num_ev mismatch: in_buf has {:d}, out_buf has {:d}",
                    in_desc->get_num_ev(), out_desc->get_num_ev());
    }

    // Get product lists for input and output from frame descriptors
    const auto& in_prods = in_desc->get_product_list();
    const auto& out_prods = out_desc->get_product_list();

    // Build a map from (input_a, input_b) -> index for the input products
    // Use a map with key = (input_a << 16) | input_b for fast lookup
    std::map<uint32_t, size_t> in_prod_to_index;
    for (size_t i = 0; i < in_prods.size(); ++i) {
        uint32_t key = (static_cast<uint32_t>(in_prods[i].input_a) << 16)
                       | static_cast<uint32_t>(in_prods[i].input_b);
        in_prod_to_index[key] = i;
    }

    // Build the index mapping from output products to input products
    prod_index_map.reserve(out_prods.size());
    for (size_t out_idx = 0; out_idx < out_prods.size(); ++out_idx) {
        uint32_t key = (static_cast<uint32_t>(out_prods[out_idx].input_a) << 16)
                       | static_cast<uint32_t>(out_prods[out_idx].input_b);
        auto it = in_prod_to_index.find(key);
        if (it == in_prod_to_index.end()) {
            FATAL_ERROR("N2Subset: output product ({:d}, {:d}) at index {:d} not found in input "
                        "buffer products",
                        out_prods[out_idx].input_a, out_prods[out_idx].input_b, out_idx);
        }
        prod_index_map.push_back(it->second);
    }

    INFO("N2Subset: mapping {:d} input products to {:d} output products", in_prods.size(),
         out_prods.size());
}


void N2Subset::main_thread() {

    N2::frameID input_frame_id(in_buf);
    N2::frameID output_frame_id(out_buf);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (in_buf->wait_for_full_frame(unique_name, input_frame_id) == nullptr) {
            break;
        }

        // Wait for the output buffer frame to be free
        if (out_buf->wait_for_empty_frame(unique_name, output_frame_id) == nullptr) {
            break;
        }

        // Allocate metadata for output frame
        out_buf->allocate_new_metadata_object(output_frame_id);

        // Create frame views
        N2FrameView input_vis(in_buf, input_frame_id);
        N2FrameView output_vis(out_buf, output_frame_id);

        // Copy metadata
        output_vis._metadata->deepCopy(input_vis._metadata);

        // Copy subset of visibilities and weights using the index mapping
        for (size_t out_idx = 0; out_idx < prod_index_map.size(); ++out_idx) {
            size_t in_idx = prod_index_map[out_idx];
            output_vis.vis[out_idx] = input_vis.vis[in_idx];
            output_vis.weight[out_idx] = input_vis.weight[in_idx];
        }

        // Copy the non-product fields (flags, eval, evec, emethod, erms, gain)
        // These have the same size in input and output (num_elements and num_ev are validated)
        output_vis.copy_data(input_vis, {N2Field::vis, N2Field::weight});

        // Mark the buffers and move on
        out_buf->mark_frame_full(unique_name, output_frame_id++);
        in_buf->mark_frame_empty(unique_name, input_frame_id++);
    }
}
