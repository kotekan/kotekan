#include "N2Subset.hpp"

#include <gsl-lite.hpp>         // for span
#include <map>                  // for map, operator==, _Rb_tree_iterator
#include <utility>              // for pair
#include <complex>              // for complex
#include <functional>           // for bind, function
#include <set>                  // for set

#include "Config.hpp"           // for Config
#include "N2FrameDesc.hpp"      // for N2Field, N2FrameDesc
#include "N2FrameView.hpp"      // for N2FrameView
#include "N2Util.hpp"           // for prod_ctype, frameID, modulo
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "kotekanLogging.hpp"   // for FATAL_ERROR, INFO
#include "fmt.hpp"              // for compile_string_to_view
#include "FrameDesc.hpp"        // for FrameDesc
#include "N2Metadata.hpp"       // for N2Metadata


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
    auto in_frame_desc = in_buf->get_frame_desc();
    if (!in_frame_desc) {
        FATAL_ERROR("N2Subset: in_buf does not have a frame descriptor set");
    }
    in_desc = std::dynamic_pointer_cast<const N2FrameDesc>(in_frame_desc);
    if (!in_desc) {
        FATAL_ERROR("N2Subset: in_buf does not have an N2FrameDesc");
    }

    auto out_frame_desc = out_buf->get_frame_desc();
    if (!out_frame_desc) {
        FATAL_ERROR("N2Subset: out_buf does not have a frame descriptor set");
    }
    out_desc = std::dynamic_pointer_cast<const N2FrameDesc>(out_frame_desc);
    if (!out_desc) {
        FATAL_ERROR("N2Subset: out_buf does not have an N2FrameDesc");
    }

    // Store num_elements from descriptors
    _in_num_elements = in_desc->get_num_elements();
    _out_num_elements = out_desc->get_num_elements();

    // Validate that output num_elements <= input (we can subset elements)
    if (_out_num_elements > _in_num_elements) {
        FATAL_ERROR("N2Subset: output num_elements ({:d}) cannot exceed input ({:d})",
                    _out_num_elements, _in_num_elements);
    }
    // Validate num_ev matches (eigenvector fields must have same size if present)
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

    INFO("N2Subset: mapping {:d} input products ({:d} elements) to {:d} output products ({:d} "
         "elements)",
         in_prods.size(), _in_num_elements, out_prods.size(), _out_num_elements);
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

        // Copy non-product fields
        if (_in_num_elements == _out_num_elements) {
            // Same num_elements: copy all non-product fields directly
            output_vis.copy_data(input_vis, {N2Field::vis, N2Field::weight});
        } else {
            // Different num_elements: copy per-element fields (flags, gain) manually,
            // extracting only the first _out_num_elements entries.
            // eval, emethod, erms are independent of num_elements and can be copied.
            output_vis.copy_data(input_vis, {N2Field::vis, N2Field::weight, N2Field::flags,
                                             N2Field::gain, N2Field::evec, N2Field::mask});

            // Copy first _out_num_elements of flags and gain
            for (uint32_t i = 0; i < _out_num_elements; ++i) {
                output_vis.flags[i] = input_vis.flags[i];
                output_vis.gain[i] = input_vis.gain[i];
                output_vis.mask[i] = input_vis.mask[i];
            }

            // Copy evec if present (num_ev > 0): each eigenvector has num_elements entries
            uint32_t num_ev = out_desc->get_num_ev();
            for (uint32_t ev = 0; ev < num_ev; ++ev) {
                for (uint32_t i = 0; i < _out_num_elements; ++i) {
                    output_vis.evec[ev * _out_num_elements + i] =
                        input_vis.evec[ev * _in_num_elements + i];
                }
            }
        }

        // Mark the buffers and move on
        out_buf->mark_frame_full(unique_name, output_frame_id++);
        in_buf->mark_frame_empty(unique_name, input_frame_id++);
    }
}
