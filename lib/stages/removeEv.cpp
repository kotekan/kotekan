#include "removeEv.hpp"

#include "Config.hpp"          // for Config
#include "Hash.hpp"            // for operator<
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for allocate_new_metadata_object, mark_frame_empty, mark_fram...
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t, datasetManager, state_id_t
#include "datasetState.hpp"    // for eigenvalueState
#include "visBuffer.hpp"       // for VisField, VisFrameView, VisField::erms, VisField::eval
#include "visUtil.hpp"         // for frameID, modulo

#include <atomic>     // for atomic_bool
#include <functional> // for _Bind_helper<>::type, bind, function
#include <utility>    // for pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(removeEv);

removeEv::removeEv(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&removeEv::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Create the state describing the eigenvalues
    auto& dm = datasetManager::instance();
    ev_state_id = dm.create_state<eigenvalueState>(0).first;
}


void removeEv::change_dataset_state(dset_id_t input_dset_id) {
    auto& dm = datasetManager::instance();
    dset_id_map[input_dset_id] = dm.add_dataset(ev_state_id, input_dset_id);
}


void removeEv::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id) == nullptr) {
            break;
        }
        auto input_frame = VisFrameView(in_buf, in_frame_id);

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id) == nullptr) {
            break;
        }

        allocate_new_metadata_object(out_buf, out_frame_id);
        VisFrameView::set_metadata((VisMetadata*)out_buf->metadata[out_frame_id]->metadata,
                                   input_frame.num_elements, input_frame.num_prod, 0);

        auto output_frame = VisFrameView(out_buf, out_frame_id);

        // check if the input dataset has changed
        if (dset_id_map.count(input_frame.dataset_id) == 0) {
            change_dataset_state(input_frame.dataset_id);
        }

        // Copy over metadata and data, but skip all ev members which may not be
        // defined
        output_frame.copy_metadata(input_frame);
        output_frame.copy_data(input_frame, {VisField::eval, VisField::evec, VisField::erms});
        output_frame.dataset_id = dset_id_map.at(input_frame.dataset_id);

        // Finish up iteration.
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);
    }
}
