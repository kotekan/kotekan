#include "bufferBadInputs.hpp"

#include "Config.hpp"         // for Config
#include "NDArray.hpp"        // for NDArray, GenericNDArray
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"      // for Telescope, station_id_t
#include "buffer.hpp"         // for Buffer
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for DEBUG, ERROR, FATAL_ERROR

#include <algorithm>  // for copy_n, fill
#include <cstddef>    // for size_t
#include <cstdint>    // for int64_t
#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <json.hpp>   // for json
#include <memory>     // for shared_ptr
#include <mutex>      // for lock_guard, mutex
#include <stdint.h>   // for uint8_t

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(bufferBadInputs);


bufferBadInputs::bufferBadInputs(Config& config_, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&bufferBadInputs::main_thread, this)) {

    num_elements = config.get<size_t>(unique_name, "num_elements");
    num_polarizations = config.get<int>(unique_name, "num_polarizations");
    num_dishes = config.get<int>(unique_name, "num_dishes");
    bf_mask_lifetime_in_samples =
        config.get<std::int64_t>(unique_name, "bf_mask_lifetime_in_samples");

    // The mask is written as a flat array in CHIME beamformer order, where element
    // `beamformer_idx` is `polarization * num_dishes + dish`. Describing it as
    // [1, num_polarizations, num_dishes] is thus only a reinterpretation, and it is valid only
    // if the shape covers the whole mask.
    if (num_elements != std::size_t(num_polarizations) * std::size_t(num_dishes))
        FATAL_ERROR("num_elements {:d} must equal num_polarizations {:d} * num_dishes {:d}",
                    num_elements, num_polarizations, num_dishes);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    // The buffer must be declared in the config; this checks that we agree with it. The buffer
    // factory attaches the configured descriptor before any stage is constructed.
    out_buf->require_frame_desc(kotekan::NDArray<std::int8_t, 3>::describe(
        "bf_mask", {1, num_polarizations, num_dishes}, {"Tbf", "P", "D"},
        {bf_mask_lifetime_in_samples, 1, 1}));

    // Construct the cylinder -> beamformer reorder table.
    // reorder[beamformer_idx] = cylinder_idx;
    reorder.resize(num_elements);

    // initialize the mask (1 == good)
    input_mask = std::vector<uint8_t>(num_elements, 1u);
    num_bad_inputs = 0;

    const Telescope& tel = Telescope::instance();

    for (size_t beamformer_idx = 0; beamformer_idx < num_elements; ++beamformer_idx) {
        station_id_t st_id =
            tel.element_index_to_station_id(beamformer_idx, ElementOrder::CHIMEBeamformer);
        reorder.at(beamformer_idx) =
            tel.station_id_to_element_index(st_id, ElementOrder::CHIMECylinder);
    }

    // Listen for bad input list updates
    std::string badInputs = config.get<std::string>(unique_name, "updatable_config/bad_inputs");
    configUpdater::instance().subscribe(
        badInputs,
        std::bind(&bufferBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));
}

bufferBadInputs::~bufferBadInputs() {}

bool bufferBadInputs::update_bad_inputs_callback(nlohmann::json& json) {
    INFO("update_bad_inputs_callback(): Received update to bad inputs list.");

    try {
        bad_inputs = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list:\n{:s}", e.what());
        return false;
    }

    // validate all inputs before changing the mask
    std::vector<bool> is_present(num_elements, false);
    for (int element : bad_inputs) {
        if (element >= (int)num_elements || element < 0) {
            ERROR("Received input with invalid index: {:d}", element);
            return false;
        }
        if (is_present.at(element)) {
            ERROR("Received input with duplicate index: {:d}", element);
            return false;
        }
        is_present.at(element) = true;
    }

    // hold lock for the entire update
    std::lock_guard<std::mutex> lock(mtx);

    // Reset the mask (1 == good)
    std::fill(input_mask.begin(), input_mask.end(), 1);

    // now update the mask
    for (int element : bad_inputs) {
        input_mask.at(reorder[element]) = 0;
    }
    num_bad_inputs = bad_inputs.size();

    DEBUG("update_bad_inputs_callback(): Bad inputs reordered and buffered.");

    return true;
}

void bufferBadInputs::main_thread() {
    const std::shared_ptr<const kotekan::GenericNDArray> frame_desc =
        out_buf->get_frame_desc<kotekan::GenericNDArray>();
    size_t nbad; // copy num bad inputs for access outside lock

    // `frame_index` counts all frames produced, not just the current slot, because the FPGA
    // sequence number has to keep increasing.
    for (std::int64_t frame_index = 0; !stop_thread; ++frame_index) {
        const int frame_id = frame_index % out_buf->num_frames;

        // get an output frame
        uint8_t* out_frame = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (out_frame == nullptr) {
            return;
        }

        // Copy from the permanent buffer
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::copy_n(input_mask.begin(), num_elements, out_frame);
            nbad = num_bad_inputs;
        }

        // Set metadata and release
        out_buf->allocate_new_metadata_object(frame_id);
        const std::shared_ptr<chordMetadata> meta = get_chord_metadata(out_buf, frame_id);
        meta->set_from_frame_desc(frame_desc);
        // Each frame is one bad feed mask sample, and each sample is valid for
        // `bf_mask_lifetime_in_samples` FPGA samples.
        meta->set_fpga_seq_num(frame_index * bf_mask_lifetime_in_samples);
        meta->set_time_downsampling_fpga(bf_mask_lifetime_in_samples);
        meta->set_rfi_num_bad_inputs(nbad);
        out_buf->mark_frame_full(unique_name, frame_id);
    }
}
