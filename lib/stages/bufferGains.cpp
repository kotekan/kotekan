#include "bufferGains.hpp"

#include "NDArray.hpp"        // for NDArray, GenericNDArray
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "chordMetadata.hpp"  // for get_chord_metadata, chordMetadata
#include "configUpdater.hpp"  // for configUpdater
#include "kotekanLogging.hpp" // for FATAL_ERROR, INFO, WARN

#include "fmt.hpp"      // for format, fmt
#include "jsonUtil.hpp" // for json_base64_to_array

#include <algorithm>  // for copy_n, fill, find
#include <chrono>     // for milliseconds
#include <exception>  // for exception
#include <functional> // for bind, placeholders
#include <memory>     // for shared_ptr
#include <stdexcept>  // for invalid_argument
#include <utility>    // for move

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(bufferGains);


bufferGains::bufferGains(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&bufferGains::main_thread, this)),
    updates_counter(
        Metrics::instance().add_counter("kotekan_buffergains_update_count", unique_name)),
    invalid_updates_counter(
        Metrics::instance().add_counter("kotekan_buffergains_invalid_update_count", unique_name)) {

    num_local_freq = config.get<int>(unique_name, "num_local_freq");
    num_polarizations = config.get<int>(unique_name, "num_polarizations");
    num_dishes = config.get<int>(unique_name, "num_dishes");
    if (num_local_freq <= 0 || num_polarizations <= 0 || num_dishes <= 0)
        FATAL_ERROR("num_local_freq ({:d}), num_polarizations ({:d}) and num_dishes ({:d}) must "
                    "be positive",
                    num_local_freq, num_polarizations, num_dishes);
    num_elements = size_t(num_polarizations) * size_t(num_dishes);

    const std::vector<float> default_gains =
        config.get_default<std::vector<float>>(unique_name, "default_gains", {1.0f, 0.0f});
    if (default_gains.size() != 2)
        FATAL_ERROR("default_gains must be [real, imag]");
    default_gain = {default_gains[0], default_gains[1]};

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    // The buffer must be declared in the config; this checks that we agree with it.
    out_buf->require_frame_desc(kotekan::NDArray<float, 4>::describe(
        "gain", {num_local_freq, num_polarizations, num_dishes, 2}, {"F", "P", "D", "C"},
        {1, 1, 1, 1}));

    metadata_source = get_buffer("metadata_source");
    metadata_source->register_consumer(unique_name);

    // The initial config block arrives through this callback during subscribe().
    configUpdater::instance().subscribe(
        config.get<std::string>(unique_name, "updatable_config/gains"),
        std::bind(&bufferGains::receive_gains, this, std::placeholders::_1));

    // From here on a rejected update must not stop kotekan; see the callback.
    initialised = true;
}

bufferGains::gainUpdate bufferGains::parse_update(const nlohmann::json& json) const {
    gainUpdate update;
    if (json.contains("update_id")) {
        if (!json.at("update_id").is_string())
            throw std::invalid_argument("bad value 'update_id': " + json.at("update_id").dump());
        update.update_id = json.at("update_id").get<std::string>();
    }

    const nlohmann::json& gain = json.at("gain");
    if (!gain.is_object())
        throw std::invalid_argument("'gain' must be an object, not " + gain.dump());
    if (gain.empty())
        return update;

    update.freq_id = json.at("freq_id").get<std::vector<int>>();
    if (gain.at("dtype") != "complex64")
        throw std::invalid_argument("'gain' dtype must be \"complex64\", not "
                                    + gain.at("dtype").dump());
    if (gain.contains("encoding") && gain.at("encoding") != "base64")
        throw std::invalid_argument("'gain' encoding must be \"base64\", not "
                                    + gain.at("encoding").dump());
    auto [data, shape] = json_base64_to_array<std::complex<float>>(gain);
    if (shape.size() != 2 || shape[0] != update.freq_id.size() || shape[1] != num_elements)
        throw std::invalid_argument(
            fmt::format(fmt("'gain' shape must be [{:d}, {:d}] (freq_id length, num_elements), "
                            "not {:s}"),
                        update.freq_id.size(), num_elements, gain.at("shape").dump()));
    update.gain = std::move(data);
    return update;
}

std::optional<int> bufferGains::missing_freq(const gainUpdate& update,
                                             const std::vector<int>& coarse_freq) {
    for (int f : coarse_freq)
        if (std::find(update.freq_id.begin(), update.freq_id.end(), f) == update.freq_id.end())
            return f;
    return std::nullopt;
}

bool bufferGains::receive_gains(nlohmann::json& json) {
    // Returning false from a configUpdater callback stops kotekan, so it is reserved for a
    // malformed initial config block; a bad update POSTed to a running correlator is logged,
    // counted and ignored instead, and the previous gains stay in force.
    const bool fatal_on_error = !initialised;

    gainUpdate update;
    try {
        update = parse_update(json);
    } catch (const std::exception& e) {
        WARN("Ignoring gains update: {:s}", e.what());
        invalid_updates_counter.inc();
        return !fatal_on_error;
    }
    update.initial = fatal_on_error;

    std::lock_guard<std::mutex> lock(mtx);
    // The frequencies are known only once main_thread has read them; before that the update
    // is checked when it is written.
    if (!update.gain.empty() && !coarse_freq.empty()) {
        if (const std::optional<int> f = missing_freq(update, coarse_freq)) {
            WARN("Ignoring gains update '{:s}': it has no gains for frequency id {:d}",
                 update.update_id, *f);
            invalid_updates_counter.inc();
            return !fatal_on_error;
        }
    }
    if (pending)
        INFO("Gains update '{:s}' supersedes '{:s}', which was not written yet", update.update_id,
             pending->update_id);
    INFO("Queued gains update '{:s}' ({:s})", update.update_id,
         update.gain.empty() ? "default gains"
                             : fmt::format(fmt("{:d} frequencies"), update.freq_id.size()));
    pending = std::move(update);
    pending_cv.notify_one();
    return true;
}

void bufferGains::main_thread() {
    // Always present: the constructor requires it.
    const std::shared_ptr<const kotekan::GenericNDArray> frame_desc =
        out_buf->get_frame_desc<kotekan::GenericNDArray>();

    // This instance's frequencies, from the first frame of the clock buffer.
    if (metadata_source->wait_for_full_frame(unique_name, 0) == nullptr)
        return;
    {
        const std::shared_ptr<const chordMetadata> clock_meta =
            get_chord_metadata(metadata_source, 0);
        if (!clock_meta->has_coarse_freq())
            FATAL_ERROR("metadata_source {:s} has no coarse_freq, needed to select the gains",
                        metadata_source->buffer_name);
        std::lock_guard<std::mutex> lock(mtx);
        coarse_freq = clock_meta->get_coarse_freq();
        if (coarse_freq.size() != size_t(num_local_freq))
            FATAL_ERROR("metadata_source {:s} has {:d} frequencies, but num_local_freq is {:d}",
                        metadata_source->buffer_name, coarse_freq.size(), num_local_freq);
    }
    metadata_source->mark_frame_empty(unique_name, 0);
    // Only the first frame is needed; stop being a consumer so that the producer does not
    // wait for us on the frames after it.
    metadata_source->unregister_consumer(unique_name);
    INFO("Gains frequencies taken from the first frame of {:s}", metadata_source->buffer_name);

    for (int frame_index = 0; !stop_thread;) {
        gainUpdate update;
        {
            std::unique_lock<std::mutex> lock(mtx);
            // Nothing wakes this wait at shutdown, so poll stop_thread.
            if (!pending_cv.wait_for(lock, std::chrono::milliseconds(100),
                                     [&] { return pending.has_value(); }))
                continue;
            update = std::move(*pending);
            pending.reset();
        }

        // An update queued before the frequencies were known is checked here. The initial
        // block is config, so a bad one is fatal.
        if (!update.gain.empty()) {
            if (const std::optional<int> f = missing_freq(update, coarse_freq)) {
                if (update.initial)
                    FATAL_ERROR("The initial gains have no entry for frequency id {:d}", *f);
                WARN("Ignoring gains update '{:s}': it has no gains for frequency id {:d}",
                     update.update_id, *f);
                invalid_updates_counter.inc();
                continue;
            }
        }

        const int frame_id = frame_index % out_buf->num_frames;
        // Blocking here paces production to the consumers; null means shutdown.
        std::complex<float>* const frame = static_cast<std::complex<float>*>(
            static_cast<void*>(out_buf->wait_for_empty_frame(unique_name, frame_id)));
        if (frame == nullptr)
            return;

        for (int f = 0; f < num_local_freq; ++f) {
            std::complex<float>* const row = frame + size_t(f) * num_elements;
            if (update.gain.empty()) {
                std::fill(row, row + num_elements, default_gain);
            } else {
                const size_t src =
                    std::find(update.freq_id.begin(), update.freq_id.end(), coarse_freq[f])
                    - update.freq_id.begin();
                std::copy_n(update.gain.data() + src * num_elements, num_elements, row);
            }
        }

        out_buf->allocate_new_metadata_object(frame_id);
        const std::shared_ptr<chordMetadata> meta = get_chord_metadata(out_buf, frame_id);
        meta->set_from_frame_desc(frame_desc);
        meta->set_coarse_freq(coarse_freq);
        out_buf->mark_frame_full(unique_name, frame_id);
        ++frame_index;
        updates_counter.inc();
        INFO("Wrote gains update '{:s}' to {:s}[{:d}]", update.update_id, out_buf->buffer_name,
             frame_id);
    }
}
