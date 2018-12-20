#include "removeEv.hpp"

#include "datasetManager.hpp"
#include "errors.h"
#include "visBuffer.hpp"
#include "visUtil.hpp"


REGISTER_KOTEKAN_PROCESS(removeEv);

removeEv::removeEv(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&removeEv::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Create the state describing the eigenvalues
    auto& dm = datasetManager::instance();
    state_uptr ev_state = std::make_unique<eigenvalueState>(0);
    ev_state_id = dm.add_state(std::move(ev_state)).first;
}


dset_id_t removeEv::change_dataset_state(dset_id_t input_dset_id) {
    auto& dm = datasetManager::instance();
    return dm.add_dataset(input_dset_id, ev_state_id);
}


void removeEv::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    dset_id_t _output_dset_id = 0;

    while (!stop_thread) {

        // Get input visibilities. We assume the shape of these doesn't change.
        if (wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id) == nullptr) {
            break;
        }
        auto input_frame = visFrameView(in_buf, in_frame_id);

        // Get output buffer for visibilities. Essentially identical to input buffers.
        if (wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(out_buf, out_frame_id);
        auto output_frame =
            visFrameView(out_buf, out_frame_id, input_frame.num_elements, input_frame.num_prod, 0);

        // check if the input dataset has changed
        if (input_dset_id != input_frame.dataset_id) {
            input_dset_id = input_frame.dataset_id;
            _output_dset_id = change_dataset_state(input_dset_id);
        }

        // Copy over metadata and data, but skip all ev members which may not be
        // defined
        output_frame.copy_metadata(input_frame);
        output_frame.copy_data(input_frame, {visField::eval, visField::evec, visField::erms});
        output_frame.dataset_id = _output_dset_id;

        // Finish up iteration.
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);
    }
}
