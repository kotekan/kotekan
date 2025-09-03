#include "InputSubset.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for operator!=, operator<, operator==, Hash
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for allocate_new_metadata_object, mark_frame_empty, mark_fram...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for state_id_t, dset_id_t, datasetManager, fingerprint_t
#include "datasetState.hpp"    // for inputState, prodState, stackState
#include "kotekanLogging.hpp"  // for FATAL_ERROR, WARN
#include "visBuffer.hpp"       // for VisFrameView, VisField, VisField::evec, VisField::flags
#include "visUtil.hpp"         // for prod_ctype, input_ctype, frameID, cfloat, modulo

#include "gsl-lite.hpp" // for span

#include <algorithm>  // for max, copy
#include <atomic>     // for atomic_bool
#include <complex>    // for complex
#include <cstdint>    // for uint32_t, uint16_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stddef.h>   // for size_t
#include <stdexcept>  // for out_of_range
#include <utility>    // for pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(InputSubset);

InputSubset::InputSubset(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&InputSubset::main_thread, this)),
    _inputs(config.get<std::set<uint32_t>>(unique_name, "inputs")) {

    // Get buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
}

void InputSubset::change_dataset_state(dset_id_t ds_id) {

    auto& dm = datasetManager::instance();

    auto frame_fingerprint = dm.fingerprint(ds_id, {"inputs", "products", "stack"});

    // If we enter here, that means we are processing the first frame, so we
    // need to calculate the actual input and visibility indices we should copy
    // out
    if (fingerprint == fingerprint_t::null) {

        // Ensure that data has not been stacked, as we can't feasibly select
        // inputs from this
        auto stack_ptr = dm.dataset_state<stackState>(ds_id);
        if (stack_ptr != nullptr) {
            FATAL_ERROR("Can not apply InputSubset data to stacked data.");
            return;
        }

        // Get the states we need.
        auto input_ptr = dm.dataset_state<inputState>(ds_id);
        auto prod_ptr = dm.dataset_state<prodState>(ds_id);
        if (input_ptr == nullptr || prod_ptr == nullptr) {
            FATAL_ERROR("Both inputState and prodState are required to subset the inputs.");
            return;
        }

        // Extract the indices of the inputs that we want to subset, and create
        // a map from the original indices, to their index in the new subsetted
        // input axis (we need this to rewrite the product axis)
        size_t ind = 0;
        uint16_t map_ind = 0;
        std::map<uint16_t, uint16_t> input_ind_map;
        std::vector<input_ctype> input_index_subset;
        for (const auto& input : input_ptr->get_inputs()) {
            if (_inputs.count(input.chan_id)) {
                input_ind.push_back(ind);
                input_index_subset.push_back(input);
                input_ind_map[ind] = map_ind;
                map_ind++;
            }
            ind++;
        }

        // Identify products where both inputs are in the subset, and rewrite
        // the product axis to contain only the matching products with their
        // indices remapped
        ind = 0;
        std::vector<prod_ctype> prod_index_subset;
        for (const auto& [input_a, input_b] : prod_ptr->get_prods()) {
            if (input_ind_map.count(input_a) && input_ind_map.count(input_b)) {
                prod_ind.push_back(ind);
                prod_index_subset.push_back({input_ind_map.at(input_a), input_ind_map.at(input_b)});
            }
            ind++;
        }

        auto isize = input_index_subset.size();
        if (isize != _inputs.size()) {
            WARN("Only found {} out of the requested {} inputs in the incoming stream.",
                 input_index_subset.size(), _inputs.size());
        }
        if (prod_index_subset.size() != (isize * (isize + 1) / 2)) {
            WARN("The incoming stream did not have the full triangle of products. "
                 "Found {} out of the requested {} products.",
                 prod_index_subset.size(), isize * (isize + 1) / 2);
        }

        states.push_back(dm.create_state<inputState>(input_index_subset).first);
        states.push_back(dm.create_state<prodState>(prod_index_subset).first);

        fingerprint = frame_fingerprint;
    }

    if (frame_fingerprint != fingerprint) {
        FATAL_ERROR("Incoming stream has changed input structure. Expected fingerprint {}, "
                    "received fingerprint {}.",
                    fingerprint, frame_fingerprint);
        return;
    }

    // If we got here, all is well, so we just register the new ID
    dset_id_map[ds_id] = dm.add_dataset(states, ds_id);
}

void InputSubset::main_thread() {

    frameID input_frame_id(in_buf);
    frameID output_frame_id(out_buf);

    while (!stop_thread) {

        // Wait for the input buffer to be filled with data
        if (wait_for_full_frame(in_buf, unique_name.c_str(), input_frame_id) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto input_frame = VisFrameView(in_buf, input_frame_id);

        // check if the input dataset has changed
        if (dset_id_map.count(input_frame.dataset_id) == 0) {
            change_dataset_state(input_frame.dataset_id);
        }

        // Wait for the output buffer frame to be free
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), output_frame_id) == nullptr) {
            break;
        }

        // Create view to output frame
        auto output_frame = VisFrameView::create_frame_view(
            out_buf, output_frame_id, input_ind.size(), prod_ind.size(), input_frame.num_ev);

        // Copy over subset of visibility shaped data
        for (size_t i = 0; i < prod_ind.size(); i++) {
            output_frame.vis[i] = input_frame.vis[prod_ind[i]];
            output_frame.weight[i] = input_frame.weight[prod_ind[i]];
        }

        // Copy over subset of input shaped data
        size_t isize = input_ind.size();
        for (size_t i = 0; i < isize; i++) {
            size_t ii = input_ind[i];
            output_frame.gain[i] = input_frame.vis[ii];
            output_frame.flags[i] = input_frame.flags[ii];

            for (size_t j = 0; j < input_frame.num_ev; j++) {
                output_frame.evec[j * isize + i] =
                    input_frame.evec[j * input_frame.num_elements + ii];
            }
        }

        // Copy metadata
        output_frame.copy_metadata(input_frame);
        output_frame.dataset_id = dset_id_map.at(input_frame.dataset_id);

        // Copy the non-input and visibility shaped parts of the buffer
        output_frame.copy_data(input_frame, {VisField::vis, VisField::weight, VisField::gain,
                                             VisField::flags, VisField::evec});

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id++);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id++);
    }
}
