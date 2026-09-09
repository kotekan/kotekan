#include "bufferBadInputs.hpp"

#include "Config.hpp"            // for Config
#include "NDArray.hpp"           // for NDArray, GenericNDArray
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"         // for Telescope, station_id_t
#include "buffer.hpp"            // for Buffer
#include "chordMetadata.hpp"     // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"     // for configUpdater
#include "kotekanLogging.hpp"    // for DEBUG, ERROR, FATAL_ERROR, INFO, WARN
#include "prometheusMetrics.hpp" // for Metrics, Counter
#include "visUtil.hpp"           // for current_time, double_to_ts, ts_to_double

#include <exception>  // for exception
#include <functional> // for bind, function, _1
#include <json.hpp>   // for json
#include <memory>     // for shared_ptr
#include <stdexcept>  // for invalid_argument
#include <stdint.h>   // for int64_t, uint8_t
#include <time.h>     // for timespec
#include <utility>    // for move

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(bufferBadInputs);


bufferBadInputs::bufferBadInputs(Config& config_, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&bufferBadInputs::main_thread, this)),
    late_updates_counter(
        Metrics::instance().add_counter("kotekan_bufferbadinputs_late_update_count", unique_name)),
    invalid_updates_counter(Metrics::instance().add_counter(
        "kotekan_bufferbadinputs_invalid_update_count", unique_name)) {

    num_elements = config.get<size_t>(unique_name, "num_elements");
    num_polarizations = config.get<int>(unique_name, "num_polarizations");
    num_dishes = config.get<int>(unique_name, "num_dishes");
    bf_mask_lifetime_in_samples = config.get<int64_t>(unique_name, "bf_mask_lifetime_in_samples");

    // The mask is written as a flat array whose element `output_idx` is
    // `polarization * num_dishes + dish`, so describing it as
    // [1, num_polarizations, num_dishes] is only a reinterpretation, and it is valid only if
    // the shape covers the whole mask.
    if (num_elements != size_t(num_polarizations) * size_t(num_dishes))
        FATAL_ERROR("num_elements {:d} must equal num_polarizations {:d} * num_dishes {:d}",
                    num_elements, num_polarizations, num_dishes);

    // Element orders of the posted bad_inputs indices and of the produced mask.
    const ElementOrder input_order =
        config.get_default<ElementOrder>(unique_name, "input_order", ElementOrder::CHIMECylinder);
    const ElementOrder output_order = config.get_default<ElementOrder>(
        unique_name, "output_order", ElementOrder::CHIMEBeamformer);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    // The buffer must be declared in the config; this checks that we agree with it. The buffer
    // factory attaches the configured descriptor before any stage is constructed.
    out_buf->require_frame_desc(kotekan::NDArray<int8_t, 3>::describe(
        "bf_mask", {1, num_polarizations, num_dishes}, {"Tbf", "P", "D"},
        {bf_mask_lifetime_in_samples, 1, 1}));

    updates.resize(config.get_default<uint32_t>(unique_name, "num_kept_updates", 5));

    // Construct the input -> output reorder table.
    // reorder[input_idx] = output_idx;
    reorder.resize(num_elements);

    const Telescope& tel = Telescope::instance();

    if (input_order == output_order) {
        // Identity; also covers orders the telescope cannot map, like the
        // CHIME defaults on a CHORD telescope.
        for (size_t idx = 0; idx < num_elements; ++idx)
            reorder.at(idx) = idx;
    } else {
        for (size_t output_idx = 0; output_idx < num_elements; ++output_idx) {
            station_id_t st_id = tel.element_index_to_station_id(output_idx, output_order);
            reorder.at(tel.station_id_to_element_index(st_id, input_order)) = output_idx;
        }
    }

    // Listen for bad input list updates. The initial config block arrives
    // through this callback during subscribe().
    std::string badInputs = config.get<std::string>(unique_name, "updatable_config/bad_inputs");
    configUpdater::instance().subscribe(
        badInputs,
        std::bind(&bufferBadInputs::update_bad_inputs_callback, this, std::placeholders::_1));

    // From here on a rejected update must not stop kotekan; see the callback.
    initialised = true;
}

bufferBadInputs::~bufferBadInputs() {}

bool bufferBadInputs::update_bad_inputs_callback(nlohmann::json& json) {
    // Returning false from a configUpdater callback stops kotekan, so it is
    // reserved for a malformed *initial* config block; a bad update POSTed to
    // a running correlator is logged, counted and ignored instead.
    const bool fatal_on_error = !initialised;

    std::vector<int> bad_inputs;
    // A missing start_time keys the update to the present: apply immediately,
    // superseding everything earlier.
    timespec start_ts = double_to_ts(current_time());
    std::string update_id;

    try {
        bad_inputs = json.at("bad_inputs").get<std::vector<int>>();

        // start_time and update_id are optional; config blocks predating them
        // keep working.
        if (json.contains("start_time")) {
            if (!json.at("start_time").is_number())
                throw std::invalid_argument("received bad value 'start_time': "
                                            + json.at("start_time").dump());
            if (json.at("start_time").get<double>() < 0)
                throw std::invalid_argument("received negative start_time: "
                                            + json.at("start_time").dump());
            start_ts = double_to_ts(json.at("start_time").get<double>());
        }
        if (json.contains("update_id")) {
            if (!json.at("update_id").is_string())
                throw std::invalid_argument("received bad value 'update_id': "
                                            + json.at("update_id").dump());
            update_id = json.at("update_id").get<std::string>();
        }
    } catch (std::exception const& e) {
        ERROR("Failed to parse bad input update, ignoring it:\n{:s}", e.what());
        invalid_updates_counter.inc();
        return !fatal_on_error;
    }

    // validate all inputs before accepting the update
    for (int element : bad_inputs) {
        if (element >= (int)num_elements || element < 0) {
            ERROR("Received input with invalid index {:d} (num_elements {:d}), "
                  "ignoring the update.",
                  element, num_elements);
            invalid_updates_counter.inc();
            return !fatal_on_error;
        }
    }

    // An update starting before one already queued could never take effect;
    // drop it rather than hide it in the queue.
    const auto queued = updates.get_all_updates();
    if (!queued.empty() && queued.back().first > start_ts) {
        WARN("Ignoring out-of-order update '{:s}': its start time {:f} precedes the newest "
             "queued update's start time {:f}.",
             update_id, ts_to_double(start_ts), ts_to_double(queued.back().first));
        late_updates_counter.inc();
        return true;
    }

    // Build the update's mask (1 == good) in output_order.
    std::vector<uint8_t> mask(num_elements, 1u);
    for (int element : bad_inputs)
        mask[reorder[element]] = 0;

    // The seq the update takes effect at -- via the telescope, so identical on every node --
    // or -1 when it predates the run and constrains nothing. Logged for diagnostics only: the
    // frames' FPGA sequence numbers count mask samples, so there is nowhere to put it. The
    // clamp also keeps to_seq()'s unsigned conversion from wrapping.
    const Telescope& tel = Telescope::instance();
    const int64_t effective_seq = start_ts > tel.to_time(0) ? (int64_t)tel.to_seq(start_ts) : -1;

    INFO("update_bad_inputs_callback(): queued update '{:s}' ({:d} bad inputs) "
         "for start time {:f} (seq {:d}).",
         update_id, bad_inputs.size(), ts_to_double(start_ts), effective_seq);

    updates.insert(start_ts, {std::move(mask)});

    return true;
}

void bufferBadInputs::main_thread() {
    // Always present: the constructor requires it.
    const std::shared_ptr<const kotekan::GenericNDArray> frame_desc =
        out_buf->get_frame_desc<kotekan::GenericNDArray>();

    // `frame_index` counts all frames produced, not just the current slot, because the FPGA
    // sequence number has to keep increasing.
    for (int64_t frame_index = 0; !stop_thread; ++frame_index) {
        const int frame_id = frame_index % out_buf->num_frames;

        // Blocking here paces production to the consumers; null means shutdown.
        uint8_t* out_frame = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (out_frame == nullptr)
            return;

        // Sample the wall clock only now: the wait above can outlast a
        // pending update.
        const timespec ref_ts = double_to_ts(current_time());

        // Compose the frame from the active update; all-good until one applies.
        const std::shared_ptr<const badInputUpdate> update = updates.get_update(ref_ts).second;
        for (size_t el = 0; el < num_elements; ++el)
            out_frame[el] = update != nullptr ? update->mask[el] : 1u;

        // Set metadata and release
        out_buf->allocate_new_metadata_object(frame_id);
        const std::shared_ptr<chordMetadata> meta = get_chord_metadata(out_buf, frame_id);
        meta->set_from_frame_desc(frame_desc);
        // Each frame is one bad feed mask sample, and each sample is valid for
        // `bf_mask_lifetime_in_samples` FPGA samples.
        meta->set_fpga_seq_num(frame_index * bf_mask_lifetime_in_samples);
        meta->set_time_downsampling_fpga(bf_mask_lifetime_in_samples);
        out_buf->mark_frame_full(unique_name, frame_id);
    }
}
