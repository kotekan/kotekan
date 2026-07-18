#include "bufferBadInputs.hpp"

#include "CHORDTelescope.hpp" // for CHORDTelescope, dishInputFields, DishType
#include "Config.hpp"         // for Config
#include "N2Util.hpp"         // for frameID
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"      // for Telescope, station_id_t
#include "buffer.hpp"         // for Buffer
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for DEBUG, ERROR

#include <algorithm>  // for count, fill
#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <json.hpp>   // for json
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

    // Element order of the incoming bad_inputs indices and of the produced
    // mask. The defaults preserve the original CHIME behaviour (flags are
    // broadcast in cylinder order, the mask is in beamformer order); CHORD
    // sends and masks in the same [P][D] order, so both are CHORDBeamformer.
    auto input_order = config.get_default<ElementOrder>(unique_name, "input_order",
                                                        ElementOrder::CHIMECylinder);
    auto output_order = config.get_default<ElementOrder>(unique_name, "output_order",
                                                         ElementOrder::CHIMEBeamformer);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    // Construct the input -> output reorder table.
    // reorder[output_idx] = input_idx;
    reorder.resize(num_elements);

    const Telescope& tel = Telescope::instance();

    if (input_order == output_order) {
        // Identity; skip the telescope round-trip so orders the telescope
        // cannot map (and equal orders on any telescope) always work.
        for (size_t idx = 0; idx < num_elements; ++idx)
            reorder.at(idx) = idx;
    } else {
        // The key defaults are the CHIME orders; a CHORD telescope cannot map
        // them and the calls below abort with "Cannot handle element order".
        // Name the fix first.
        ElementOrder fiducial = tel.fiducial_element_order();
        if ((fiducial == ElementOrder::CHORDEarly || fiducial == ElementOrder::CHORDBeamformer)
            && (input_order == ElementOrder::CHIMECylinder
                || output_order == ElementOrder::CHIMEBeamformer))
            ERROR("bad-input element orders ({} -> {}) are the CHIME defaults; set "
                  "input_order/output_order explicitly for this telescope (fiducial order: {}).",
                  input_order, output_order, fiducial);

        for (size_t output_idx = 0; output_idx < num_elements; ++output_idx) {
            station_id_t st_id = tel.element_index_to_station_id(output_idx, output_order);
            reorder.at(output_idx) = tel.station_id_to_element_index(st_id, input_order);
        }
    }

    // Baseline mask from the telescope's dish table: elements whose dish is
    // not a real array dish (Fake or an RFI antenna) are never valid inputs
    // and stay masked independent of the posted bad-inputs list. This is what
    // inventBFMask (mode: auto) encoded before this stage replaced it.
    baseline_mask = std::vector<uint8_t>(num_elements, 1u);
    const auto* chord_tel = dynamic_cast<const CHORDTelescope*>(&tel);
    if (chord_tel != nullptr) {
        dishInputFields dish_inputs;
        chord_tel->fill_input_maps(dish_inputs);
        for (size_t el = 0; el < num_elements; ++el) {
            uint64_t dish;
            uint64_t pol;
            station_id_t st_id = tel.element_index_to_station_id(el, output_order);
            chord_tel->decode_station_id(st_id, dish, pol);
            if (dish_inputs.type.at(dish) != DishType::ArrayDish)
                baseline_mask[el] = 0;
        }
    }

    // initialize the mask (1 == good, non-array dishes already masked)
    input_mask = baseline_mask;
    num_bad_inputs = std::count(input_mask.begin(), input_mask.end(), 0u);

    // Listen for bad input list updates
    std::string badInputs = config.get<std::string>(unique_name, "updatable_config/bad_inputs");
    configUpdater::instance().subscribe(
        badInputs,
        std::bind(&bufferBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));
}

bufferBadInputs::~bufferBadInputs() {}

bool bufferBadInputs::update_bad_inputs_callback(nlohmann::json& json) {
    DEBUG("update_bad_inputs_callback(): Update to bad inputs list.");

    // hold lock for the entire update
    std::lock_guard<std::mutex> lock(mtx);

    // Reset the mask (1 == good, non-array dishes always masked)
    input_mask = baseline_mask;

    bool all_valid = true;

    try {
        bad_inputs = json["bad_inputs"].get<std::vector<int>>();
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input list:\n{:s}", e.what());
        return false;
    }

    // Add current bad input to the mask
    for (int element : bad_inputs) {
        if (element < (int)num_elements && element >= 0) {
            input_mask[reorder[element]] = 0;
        } else {
            ERROR("Got input with invalid index: {:d}", element);
            all_valid = false;
        }
    }

    num_bad_inputs = std::count(input_mask.begin(), input_mask.end(), 0u);

    DEBUG("update_bad_inputs_callback(): Bad inputs reordered and buffered.");

    return all_valid;
}

void bufferBadInputs::main_thread() {
    N2::frameID frame_id(out_buf);

    while (!stop_thread) {
        // get an output frame; a null frame means kotekan is shutting down
        uint8_t* out_frame = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (out_frame == nullptr)
            break;

        // Copy from the permanent buffer
        uint64_t frame_num_bad_inputs;
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::copy_n(input_mask.begin(), num_elements, out_frame);
            frame_num_bad_inputs = num_bad_inputs;
        }

        // Set metadata and release
        out_buf->allocate_new_metadata_object(frame_id);
        get_chord_metadata(out_buf, frame_id)->set_rfi_num_bad_inputs(frame_num_bad_inputs);
        out_buf->mark_frame_full(unique_name, frame_id);
        frame_id++;
    }
}
