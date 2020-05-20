#include "visAccumulate.hpp"

#include "Config.hpp"       // for Config
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.h"              // for register_producer, Buffer, allocate_new_metadata_object
#include "bufferContainer.hpp"   // for bufferContainer
#include "chimeMetadata.h"       // for chimeMetadata, get_fpga_seq_num, get_lost_timesamples
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for state_id_t, datasetManager, dset_id_t
#include "datasetState.hpp"      // for eigenvalueState, freqState, gatingState, inputState
#include "factory.hpp"           // for FACTORY
#include "kotekanLogging.hpp"    // for FATAL_ERROR, INFO, logLevel, DEBUG
#include "metadata.h"            // for metadataContainer
#include "prometheusMetrics.hpp" // for Counter, MetricFamily, Metrics
#include "version.h"             // for get_git_commit_hash
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for prod_ctype, frameID, input_ctype, modulo, operator+

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span<>::iterator, span
#include "json.hpp"     // for json, basic_json, iteration_proxy_value, basic_json<>...

#include <algorithm> // for max, fill, copy, copy_backward, equal, transform
#include <assert.h>  // for assert
#include <atomic>    // for atomic_bool
#include <cmath>     // for pow
#include <complex>   // for operator*, complex
#include <cstring>   // for memcpy
#include <exception> // for exception
#include <iterator>  // for back_insert_iterator, begin, end, back_inserter
#include <mutex>     // for lock_guard, mutex
#include <numeric>   // for iota
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error, invalid_argument
#include <time.h>    // for size_t, timespec
#include <tuple>     // for get
#include <vector>    // for vector, vector<>::iterator, __alloc_traits<>::value_type


using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(visAccumulate);


visAccumulate::visAccumulate(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&visAccumulate::main_thread, this)),
    skipped_frame_counter(Metrics::instance().add_counter(
        "kotekan_visaccumulate_skipped_frame_total", unique_name, {"freq_id", "reason"})) {

    // Fetch any simple configuration
    num_elements = config.get<size_t>(unique_name, "num_elements");
    num_freq_in_frame = config.get_default<size_t>(unique_name, "num_freq_in_frame", 1);
    block_size = config.get<size_t>(unique_name, "block_size");
    samples_per_data_set = config.get<size_t>(unique_name, "samples_per_data_set");
    max_age = config.get_default<float>(unique_name, "max_age", 60.0);

    // Get the indices for reordering
    auto input_reorder = parse_reorder_default(config, unique_name);
    input_remap = std::get<0>(input_reorder);

    float int_time = config.get_default<float>(unique_name, "integration_time", -1.0);

    // If the integration time was set then calculate the number of GPU frames
    // we need to integrate for.
    if (int_time >= 0.0) {
        // TODO: don't hard code the sample time length
        // TODO: CHIME specific
        float frame_length = samples_per_data_set * 2.56e-6;

        // Calculate nearest *even* number of frames
        num_gpu_frames = 2 * ((int)(int_time / frame_length) / 2);

        INFO("Integrating for {:d} gpu frames (={:.2f} s  ~{:.2f} s)", num_gpu_frames,
             frame_length * num_gpu_frames, int_time);
    } else {
        num_gpu_frames = config.get<size_t>(unique_name, "num_gpu_frames");
        INFO("Integrating for {:d} gpu frames.", num_gpu_frames);
    }

    // Get the minimum number of samples for an output frame
    float low_sample_fraction = config.get_default<float>(unique_name, "low_sample_fraction", 0.01);
    minimum_samples = (size_t)(low_sample_fraction * num_gpu_frames * samples_per_data_set);

    size_t nb = num_elements / block_size;
    num_prod_gpu = num_freq_in_frame * nb * (nb + 1) * block_size * block_size / 2;

    // Get everything we need for registering dataset states

    // --> get metadata
    std::string instrument_name =
        config.get_default<std::string>(unique_name, "instrument_name", "chime");

    std::vector<uint32_t> freq_ids;

    // Get the frequency IDs that are on this stream, check the config or just
    // assume all CHIME channels
    auto& tel = Telescope::instance();
    if (config.exists(unique_name, "freq_ids")) {
        freq_ids = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");
    } else {
        freq_ids.resize(tel.num_freq());
        std::iota(std::begin(freq_ids), std::end(freq_ids), 0);
    }

    // Create the frequency specification
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;
    std::transform(std::begin(freq_ids), std::end(freq_ids), std::back_inserter(freqs),
                   [&tel](uint32_t id) -> std::pair<uint32_t, freq_ctype> {
                       return {id, {tel.to_freq(id), tel.freq_width(id)}};
                   });

    // The input specification from the config
    std::vector<input_ctype> inputs = std::get<1>(input_reorder);

    size_t num_elements = inputs.size();

    // Create the product specification
    std::vector<prod_ctype> prods;
    prods.reserve(num_elements * (num_elements + 1) / 2);
    for (uint16_t i = 0; i < num_elements; i++) {
        for (uint16_t j = i; j < num_elements; j++) {
            prods.push_back({i, j});
        }
    }

    // get dataset ID for out frames
    base_dataset_id = base_dataset_state(instrument_name, freqs, inputs, prods);

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // Create the state for the main visibility accumulation
    gated_datasets.emplace_back(
        out_buf, gateSpec::create("uniform", "vis", kotekan::logLevel(_member_log_level)),
        num_prod_gpu);
    gated_datasets.at(0).output_dataset_id = base_dataset_id;


    // Get and validate any gating config
    nlohmann::json gating_conf = config.get_default<nlohmann::json>(unique_name, "gating", {});
    if (!gating_conf.empty() && !gating_conf.is_object()) {
        FATAL_ERROR("Gating config must be a dictionary: {:s}", gating_conf.dump());
    }

    if (!gating_conf.empty() && num_freq_in_frame > 1) {
        FATAL_ERROR("Cannot use gating with multifrequency GPU buffers[num_freq_in_frame={:d}; "
                    "gating config={:s}].",
                    num_freq_in_frame, gating_conf.dump());
    }

    // Register gating update callbacks
    std::map<std::string, std::function<bool(nlohmann::json&)>> callbacks;

    for (auto& it : gating_conf.items()) {

        // Get the name of the gated dataset
        std::string name = it.key();

        // Validate and fetch the gating mode
        try {
            if (!it.value().at("mode").is_string()) {
                throw std::invalid_argument(fmt::format(fmt("Config for gated dataset {:s} did "
                                                            "not have a valid mode argument: {:s}"),
                                                        name, it.value().dump()));
            }
        } catch (std::exception& e) {
            FATAL_ERROR("Failure reading 'mode' from config: {:s}", e.what());
        }
        std::string mode = it.value().at("mode");

        if (!FACTORY(gateSpec)::exists(mode)) {
            FATAL_ERROR("Requested gating mode {:s} for dataset {:s} is not a known.", name, mode);
        }

        INFO("Creating gated dataset {:s} of type {:s}", name, mode);

        // Validate and fetch the output buffer name
        try {
            if (!it.value().at("buf").is_string()) {
                throw std::invalid_argument(fmt::format(fmt("Config for gated dataset {:s} did "
                                                            "not have a valid buf argument: {:s}"),
                                                        name, it.value().dump()));
            }
        } catch (std::exception& e) {
            FATAL_ERROR("Failure reading 'buf' from config: {:s}", e.what());
        }
        std::string buffer_name = it.value().at("buf");

        // Fetch and register the buffer
        auto buf = buffer_container.get_buffer(buffer_name);
        register_producer(buf, unique_name.c_str());

        // Create the gated dataset and register the update callback
        gated_datasets.emplace_back(
            buf, gateSpec::create(mode, name, kotekan::logLevel(_member_log_level)), num_prod_gpu);

        auto& state = gated_datasets.back();
        callbacks[name] = [&state](nlohmann::json& json) -> bool {
            bool success = state.spec->update_spec(json);
            if (success) {
                std::lock_guard<std::mutex> lock(state.state_mtx);
                state.changed = true;
            }
            return success;
        };
    }

    configUpdater::instance().subscribe(this, callbacks);
}


dset_id_t visAccumulate::base_dataset_state(std::string& instrument_name,
                                            std::vector<std::pair<uint32_t, freq_ctype>>& freqs,
                                            std::vector<input_ctype>& inputs,
                                            std::vector<prod_ctype>& prods) {
    // weight calculation is hardcoded, so is the weight type name
    const std::string weight_type = "inverse_var";
    const std::string git_tag = get_git_commit_hash();

    // create all the states
    datasetManager& dm = datasetManager::instance();
    std::vector<state_id_t> base_states;
    base_states.push_back(dm.create_state<freqState>(freqs).first);
    base_states.push_back(dm.create_state<inputState>(inputs).first);
    base_states.push_back(dm.create_state<prodState>(prods).first);
    base_states.push_back(dm.create_state<eigenvalueState>(0).first);
    base_states.push_back(
        dm.create_state<metadataState>(weight_type, instrument_name, git_tag).first);

    // register root dataset
    return dm.add_dataset(base_states);
}


dset_id_t visAccumulate::gate_dataset_state(const gateSpec& spec) {

    // register with the datasetManager
    datasetManager& dm = datasetManager::instance();
    state_id_t gstate_id = dm.create_state<gatingState>(spec).first;

    // register gated dataset
    return dm.add_dataset(gstate_id, base_dataset_id);
}


void visAccumulate::main_thread() {

    frameID in_frame_id(in_buf);

    // Hold the gated datasets that are enabled;
    std::vector<std::reference_wrapper<internalState>> enabled_gated_datasets;

    uint32_t last_frame_count = 0;
    uint32_t frames_in_this_cycle = 0;

    // Temporary arrays for storing intermediates
    std::vector<int32_t> vis_even(2 * num_prod_gpu);
    int32_t samples_even = 0;

    auto& tel = Telescope::instance();

    // Have we initialised a frame for writing yet
    bool init = false;

    while (!stop_thread) {

        // Fetch a new frame and get its sequence id
        uint8_t* in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_frame == nullptr)
            break;

        int32_t* input = (int32_t*)in_frame;
        uint64_t frame_count = (get_fpga_seq_num(in_buf, in_frame_id) / samples_per_data_set);

        // Start and end times of this frame
        // TODO: CHIME specific
        timespec t_s = ((chimeMetadata*)in_buf->metadata[in_frame_id]->metadata)->gps_time;
        timespec t_e = add_nsec(t_s, samples_per_data_set * 2560L); // Frame length CHIME specific

        // If we have wrapped around we need to write out any frames that have
        // been filled in previous iterations. In here we need to reorder the
        // accumulates and do any final manipulations.
        bool wrapped = (last_frame_count / num_gpu_frames) < (frame_count / num_gpu_frames);

        if (init && wrapped) {

            internalState& d0 = enabled_gated_datasets.at(0);

            // Debias the weights estimate, by subtracting out the bias estimation
            float w = d0.weight_diff_sum / pow(d0.sample_weight_total, 2);
            for (size_t i = 0; i < num_prod_gpu; i++) {
                float di = d0.vis1[2 * i];
                float dr = d0.vis1[2 * i + 1];
                d0.vis2[i] -= w * (dr * dr + di * di);
            }

            // Iterate over *only* the gated datasets (remember that element
            // zero is the vis), and remove the bias and copy in the variance
            for (size_t i = 1; i < enabled_gated_datasets.size(); i++) {
                combine_gated(enabled_gated_datasets.at(i), d0);
            }

            // Finalise the output and release the frames
            for (internalState& dset : enabled_gated_datasets) {
                finalise_output(dset, t_s);
            }

            init = false;
            frames_in_this_cycle = 0;
        }

        // We've started accumulating a new frame. Initialise the output and
        // copy over any metadata.
        if (frame_count % num_gpu_frames == 0) {

            // Reset gated streams and find which ones are enabled for this period
            enabled_gated_datasets.clear();
            for (auto& state : gated_datasets) {
                if (reset_state(state, t_s)) {
                    enabled_gated_datasets.push_back(state);
                }
            }

            // For each dataset and frequency, claim an empty frame and initialise it...
            for (internalState& dset : enabled_gated_datasets) {
                // Initialise the output, if true is returned we need to exit
                // the process as kotekan is shutting down
                if (initialise_output(dset, in_frame_id)) {
                    return;
                }
            }

            init = true;
        }

        // If we've got to here and we've not initialised we need to skip this frame.
        if (init) {

            // Get the amount of data in the frame
            // TODO: for the multifrequency support this probably needs to become frequency
            // dependent
            int32_t lost_in_frame = get_lost_timesamples(in_buf, in_frame_id);
            int32_t rfi_in_frame = get_rfi_flagged_samples(in_buf, in_frame_id);

            // Assert that we haven't got an issue calculating the lost data
            // This did happen when the RFI system was messing up.
            assert(lost_in_frame >= 0);
            assert(rfi_in_frame >= 0);
            assert(samples_per_data_set >= (size_t)lost_in_frame);
            assert(samples_per_data_set >= (size_t)rfi_in_frame);

            int32_t samples_in_frame = samples_per_data_set - lost_in_frame;

            // Accumulate the weighted data into each dataset. At the moment this
            // doesn't really work if there are multiple frequencies in the same buffer..
            for (internalState& dset : enabled_gated_datasets) {

                float freq_in_MHz = tel.to_freq(dset.frames[0].freq_id);
                float w = dset.calculate_weight(t_s, t_e, freq_in_MHz);

                // Don't bother to accumulate if weight is zero
                if (w == 0)
                    break;

                // TODO: implement generalised non uniform weighting, I'm primarily
                // not doing this because I don't want to burn cycles doing the
                // multiplications
                // Perform primary accumulation (assume that the weight is one)
                for (size_t i = 0; i < 2 * num_prod_gpu; i++) {
                    dset.vis1[i] += input[i];
                }

                dset.sample_weight_total += samples_in_frame;

                for (auto& frame : dset.frames) {
                    // Accumulate the samples/RFI
                    frame.fpga_seq_total += (uint32_t)samples_in_frame;
                    frame.rfi_total += (uint32_t)rfi_in_frame;

                    DEBUG("Lost samples {:d}, RFI flagged samples {:d}, total_samples: {:d}",
                          lost_in_frame, rfi_in_frame, frame.fpga_seq_total);
                }
            }

            // We are calculating the weights by differencing even and odd samples.
            // Every even sample we save the set of visibilities...
            if (frame_count % 2 == 0) {
                std::memcpy(vis_even.data(), input, 8 * num_prod_gpu);
                samples_even = samples_in_frame;
            }
            // ... every odd sample we accumulate the squared differences into the weight dataset
            // NOTE: this incrementally calculates the variance, but eventually
            // output_frame.weight will hold the *inverse* variance
            // TODO: we might need to account for packet loss in here too, but it
            // would require some awkward rescalings
            else {
                internalState& d0 = enabled_gated_datasets.at(0); // Save into the main vis dataset
                for (size_t i = 0; i < num_prod_gpu; i++) {
                    // NOTE: avoid using the slow std::complex routines in here
                    float di = input[2 * i] - vis_even[2 * i];
                    float dr = input[2 * i + 1] - vis_even[2 * i + 1];
                    d0.vis2[i] += (dr * dr + di * di);
                }

                // Accumulate the squared samples difference which we need for
                // debiasing the variance estimate
                float samples_diff = samples_in_frame - samples_even;
                d0.weight_diff_sum += samples_diff * samples_diff;
            }
        }

        // Move the input buffer on one step
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
        last_frame_count = frame_count;
        frames_in_this_cycle++;
    }
}


bool visAccumulate::initialise_output(visAccumulate::internalState& state, int in_frame_id) {
    // TODO: CHIME
    auto metadata = (const chimeMetadata*)in_buf->metadata[in_frame_id]->metadata;

    for (size_t freq_ind = 0; freq_ind < num_freq_in_frame; freq_ind++) {

        if (wait_for_empty_frame(state.buf, unique_name.c_str(), state.frame_id + freq_ind)
            == nullptr) {
            return true;
        }

        allocate_new_metadata_object(state.buf, state.frame_id + freq_ind);
        state.frames.emplace_back(state.buf, state.frame_id + freq_ind, num_elements, 0);
        auto& frame = state.frames[freq_ind];

        // Copy over the metadata
        // TODO: CHIME
        frame.fill_chime_metadata(metadata);

        // TODO: set frequency id in some sensible generic manner. This doesn't
        // actually work for ICEboard based multifrequency systems
        frame.freq_id += freq_ind;

        // Set dataset ID produced by the dM
        // TODO: this should be different for different gated streams
        frame.dataset_id = state.output_dataset_id;

        // Set the length of time this frame will cover
        frame.fpga_seq_length = samples_per_data_set * num_gpu_frames;

        // Reset the total accumulated and RFI flagged samples
        frame.fpga_seq_total = 0;
        frame.rfi_total = 0;

        // Fill other datasets with reasonable values
        std::fill(frame.flags.begin(), frame.flags.end(), 1.0);
        std::fill(frame.evec.begin(), frame.evec.end(), 0.0);
        std::fill(frame.eval.begin(), frame.eval.end(), 0.0);
        frame.erms = 0;
        std::fill(frame.gain.begin(), frame.gain.end(), 1.0);
    }

    return false;
}


void visAccumulate::combine_gated(visAccumulate::internalState& gate,
                                  visAccumulate::internalState& vis) {
    // NOTE: getting all of these scaling right is a pain. At the moment they
    // assume that within an `on` period the weights applied are one.

    // Subtract out the bias from the gated data
    float scl = gate.sample_weight_total / vis.sample_weight_total;
    for (size_t i = 0; i < 2 * num_prod_gpu; i++) {
        gate.vis1[i] -= (int32_t)(scl * vis.vis1[i]);
    }

    // TODO: very strong assumption that the weights are one (when on) baked in
    // here.
    gate.sample_weight_total = vis.sample_weight_total - gate.sample_weight_total;

    // Copy in the proto weight data
    for (size_t i = 0; i < num_prod_gpu; i++) {
        gate.vis2[i] = scl * (1.0 - scl) * vis.vis2[i];
    }
}


void visAccumulate::finalise_output(visAccumulate::internalState& state,
                                    timespec newest_frame_time) {

    // Determine the weighting factors (if weight is zero we should just
    // multiply the visibilities by zero so as not to generate Infs)
    float w = state.sample_weight_total;
    float iw = (w != 0.0) ? (1.0 / w) : 0.0;

    bool blocked = false;

    // Loop over the frequencies in the frame and unpack the accumulates
    // into the output frame...
    for (size_t freq_ind = 0; freq_ind < num_freq_in_frame; freq_ind++) {
        auto output_frame = state.frames[freq_ind];

        // Check if we need to skip the frame.
        //
        // TODO: if we have multifrequencies, if any need to be skipped all of
        // the following ones must be too. I think this requires the buffer
        // mechanism being rewritten to fix this one.
        if (ts_to_double(std::get<1>(output_frame.time) - newest_frame_time) > max_age) {
            skipped_frame_counter.labels({std::to_string(output_frame.freq_id), "age"}).inc();
            blocked = true;
            continue;
        }
        if (output_frame.fpga_seq_total < minimum_samples) {
            skipped_frame_counter.labels({std::to_string(output_frame.freq_id), "flagged"}).inc();
            blocked = true;
            continue;
        } else if (blocked) {
            // If we are here, an earlier frame was skipped and thus we have to
            // throw this one away too. Mark it as skipped because it was
            // blocked.
            skipped_frame_counter.labels({std::to_string(output_frame.freq_id), "blocked"}).inc();
            continue;
        }

        // Copy the visibilities into place
        map_vis_triangle(input_remap, block_size, num_elements, freq_ind,
                         [&](int32_t pi, int32_t bi, bool conj) {
                             cfloat t = {(float)state.vis1[2 * bi + 1], (float)state.vis1[2 * bi]};
                             t = !conj ? t : std::conj(t);
                             output_frame.vis[pi] = iw * t;
                         });

        // Unpack and invert the weights
        map_vis_triangle(input_remap, block_size, num_elements, freq_ind,
                         [&](int32_t pi, int32_t bi, bool conj) {
                             (void)conj;
                             float t = state.vis2[bi];
                             output_frame.weight[pi] = w * w / t;
                         });

        mark_frame_full(state.buf, unique_name.c_str(), state.frame_id++);
    }
}


bool visAccumulate::reset_state(visAccumulate::internalState& state, timespec t) {

    // Reset the internal counters
    state.sample_weight_total = 0;
    state.weight_diff_sum = 0;

    // Acquire the lock so we don't get confused by any changes made via the
    // REST callback
    {
        std::lock_guard<std::mutex> lock(state.state_mtx);

        // Update the weight function in case an update arrives mid integration
        // This is done every cycle to allow the calculation to change with time
        // (without any external update), e.g. in SegmentedPolyco's.
        if (!state.spec->enabled()) {
            state.calculate_weight = nullptr;
            return false;
        }
        state.calculate_weight = state.spec->weight_function(t);


        // Update dataset ID if an external change occurred
        if (state.changed) {
            state.output_dataset_id = gate_dataset_state(*state.spec.get());
            state.changed = false;
        }
    }

    // Zero out accumulation arrays
    std::fill(state.vis1.begin(), state.vis1.end(), 0.0);
    std::fill(state.vis2.begin(), state.vis2.end(), 0.0);

    // Remove all the old frame views
    state.frames.clear();

    return true;
}


visAccumulate::internalState::internalState(Buffer* out_buf, std::unique_ptr<gateSpec> gate_spec,
                                            size_t nprod) :
    buf(out_buf),
    frame_id(buf),
    spec(std::move(gate_spec)),
    changed(true),
    vis1(2 * nprod),
    vis2(nprod) {}
