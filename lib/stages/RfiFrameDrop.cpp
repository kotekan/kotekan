#include "RfiFrameDrop.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for Hash, operator!=
#include "Stage.hpp"             // for Stage
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"         // for Telescope
#include "buffer.h"              // for mark_frame_empty, Buffer, wait_for_full_frame, get_meta...
#include "bufferContainer.hpp"   // for bufferContainer
#include "chimeMetadata.hpp"     // for chimeMetadata, get_dataset_id, get_fpga_seq_num, set_da...
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for dset_id_t, datasetManager, state_id_t
#include "kotekanLogging.hpp"    // for WARN, INFO, DEBUG, DEBUG2
#include "prometheusMetrics.hpp" // for Metrics, Counter, MetricFamily, prometheus_counter_ptr_t
#include "visUtil.hpp"           // for frameID, modulo

#include "fmt.hpp" // for format, fmt

#include <algorithm>  // for copy, fill, max
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cmath>      // for sqrt, fabs
#include <cstring>    // for memcpy
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, function, bind, _Placeholder, _1
#include <map>        // for map, map<>::mapped_type
#include <memory>     // for allocator_traits<>::value_type, __shared_ptr_access
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdint.h>   // for uint8_t, int64_t, uint32_t
#include <string>     // for string, to_string
#include <tuple>      // for tie, tuple
#include <utility>    // for pair


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(RfiFrameDrop);

RfiFrameDrop::RfiFrameDrop(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&RfiFrameDrop::main_thread, this)),
    failing_frame_counter(
        Metrics::instance().add_counter("kotekan_rfiframedrop_failing_frame_total", unique_name,
                                        {"freq_id", "threshold", "fraction"})),
    dropped_frame_counter(Metrics::instance().add_counter(
        "kotekan_rfiframedrop_dropped_frame_total", unique_name, {"freq_id"})),
    frame_counter(Metrics::instance().add_counter("kotekan_rfiframedrop_frame_total", unique_name,
                                                  {"freq_id"})) {

    _buf_in_vis = get_buffer("in_buf_vis");
    _buf_in_sk = get_buffer("in_buf_sk");
    _buf_out = get_buffer("out_buf");
    register_consumer(_buf_in_vis, unique_name.c_str());
    register_consumer(_buf_in_sk, unique_name.c_str());
    register_producer(_buf_out, unique_name.c_str());

    auto num_samples = config.get<size_t>(unique_name, "samples_per_data_set");
    sk_step = config.get<size_t>(unique_name, "sk_step");
    num_elements = config.get<size_t>(unique_name, "num_elements");
    num_sub_frames = config.get<size_t>(unique_name, "num_sub_frames");
    sk_samples_per_frame = num_samples / sk_step / num_sub_frames;
    samples_per_sub_frame = num_samples / num_sub_frames;

    // subscribe on updates on threshold and enable_rfi_zero
    std::map<std::string, std::function<bool(nlohmann::json&)>> callbacks;
    callbacks["enable"] =
        std::bind(&RfiFrameDrop::rest_enable_callback, this, std::placeholders::_1);
    callbacks["thresholds"] =
        std::bind(&RfiFrameDrop::rest_thresholds_callback, this, std::placeholders::_1);
    kotekan::configUpdater::instance().subscribe(this, callbacks);

    assert((size_t)_buf_in_sk->frame_size == sizeof(float) * sk_samples_per_frame * num_sub_frames);
}

void RfiFrameDrop::main_thread() {

    auto& tel = Telescope::instance();

    frameID frame_id_in_vis(_buf_in_vis);
    frameID frame_id_in_sk(_buf_in_sk);
    frameID frame_id_out(_buf_out);

    dset_id_t dset_id_in = dset_id_t::null;
    dset_id_t dset_id_out = dset_id_t::null;

    std::vector<float> sk_delta(sk_samples_per_frame);

    // keep track of the dataset state
    state_id_t last_state_id = state_id_t::null;

    // Lock to protect dataset state and all update data (they are all changed together in either of
    // the endpoint callbacks)
    std::unique_lock<std::mutex> lock(update_mutex, std::defer_lock);

    while (!stop_thread) {
        // Fetch the input buffers
        uint8_t* frame_in_vis =
            wait_for_full_frame(_buf_in_vis, unique_name.c_str(), frame_id_in_vis);
        float* frame_in_sk =
            (float*)wait_for_full_frame(_buf_in_sk, unique_name.c_str(), frame_id_in_sk);

        // Test to ensure we actually got valid buffers back
        if (frame_in_vis == nullptr || frame_in_sk == nullptr)
            break;

        auto* metadata_vis = (chimeMetadata*)get_metadata(_buf_in_vis, frame_id_in_vis);
        auto* metadata_sk = (chimeMetadata*)get_metadata(_buf_in_sk, frame_id_in_sk);

        // Set the frequency index from the stream id of the metadata
        uint32_t freq_id = tel.to_freq_id(_buf_in_vis, frame_id_in_vis);

        // Try and synchronize up the frames. Even though they arrive at
        // different rates, this should eventually sync them up.
        auto vis_seq = metadata_vis->fpga_seq_num;
        auto sk_seq = metadata_sk->fpga_seq_num;

        if (vis_seq < sk_seq) {
            DEBUG("Dropping incoming N2 frame to sync up. Vis frame: {}; SK frame: {}, diff {}",
                  vis_seq, sk_seq, vis_seq - sk_seq);
            mark_frame_empty(_buf_in_vis, unique_name.c_str(), frame_id_in_vis++);
            continue;
        }
        if (sk_seq < vis_seq) {
            DEBUG("Dropping incoming SK frame to sync up. Vis frame: {}; SK frame: {}, diff {}",
                  vis_seq, sk_seq, vis_seq - sk_seq);
            mark_frame_empty(_buf_in_sk, unique_name.c_str(), frame_id_in_sk++);
            continue;
        }
        DEBUG2("Frames are synced. Vis frame: {}; SK frame: {}, diff {}", vis_seq, sk_seq,
               vis_seq - sk_seq);

        // Calculate the scaling to turn kurtosis value into sigma
        size_t num_inputs = num_elements - metadata_vis->rfi_num_bad_inputs;
        float sigma_scale = sqrt((num_inputs * (sk_step - 1) * (sk_step + 2) * (sk_step + 3))
                                 / (4.0 * sk_step * sk_step));

        // Lock update mutex to not allow updates being processed during this critical section
        lock.lock();

        // Check if we need to register a new dataset
        dset_id_t dset_id_in_new = get_dataset_id(_buf_in_vis, frame_id_in_vis);
        if (dset_id_in_new != dset_id_in || state_id != last_state_id) {
            dset_id_out = dm.add_dataset(state_id, dset_id_in_new);
            dset_id_in = dset_id_in_new;
            last_state_id = state_id;
        }

        // pair<float, float> of thresholds and fractions
        const auto thresholds = state_ptr->get_thresholds();

        // copy data while lock held
        bool enabled_copy = state_ptr->get_enabled();
        std::vector<size_t> sk_exceeds_copy(sk_exceeds.size());
        std::copy(sk_exceeds.begin(), sk_exceeds.end(), sk_exceeds_copy.begin());

        // Release the update lock
        lock.unlock();


        for (size_t ii = 0; ii < num_sub_frames; ii++) {

            if (wait_for_full_frame(_buf_in_vis, unique_name.c_str(), frame_id_in_vis) == nullptr) {
                break;
            }

            auto sf_seq = get_fpga_seq_num(_buf_in_vis, frame_id_in_vis);

            // Check that we are still synchronized with the frame we are
            // expecting. If not (and this may happen if the Valve process is
            // active), we will just skip the set of sub frames and hopefully we
            // will resync
            if (sf_seq != (int64_t)(sk_seq + ii * samples_per_sub_frame)) {
                DEBUG("Lost synchronization. Dropping data and resetting.");
                mark_frame_empty(_buf_in_vis, unique_name.c_str(), frame_id_in_vis++);
                break;
            }

            bool skip = false;


            // Process all the SK values to their deltas and for each threshold
            // we need to count how many samples exceed that threshold
            for (size_t jj = 0; jj < sk_samples_per_frame; jj++) {
                float sk_sig =
                    fabs(sigma_scale * (frame_in_sk[ii * sk_samples_per_frame + jj] - 1.0f));

                for (size_t kk = 0; kk < thresholds.size(); kk++)
                    sk_exceeds_copy.at(kk) += (sk_sig > thresholds.at(kk).first);
            }

            for (size_t kk = 0; kk < thresholds.size(); kk++) {
                if (sk_exceeds_copy.at(kk) > num_sk.at(kk)) {
                    skip = true;
                    failing_frame_counter
                        ->labels({std::to_string(freq_id), std::to_string(thresholds.at(kk).first),
                                  std::to_string(thresholds.at(kk).second)})
                        .inc();
                }
                // Reset counters for the next sub_frame
                sk_exceeds_copy.at(kk) = 0;
            }

            // If no frame exceeded it's threshold then we should transfer the
            // frame over to the output and release it. If we want to drop the
            // incoming frame then we leave the output as is.
            if (!skip || !enabled_copy) {

                if (wait_for_empty_frame(_buf_out, unique_name.c_str(), frame_id_out) == nullptr) {
                    break;
                }
                copy_frame(_buf_in_vis, frame_id_in_vis, _buf_out, frame_id_out);
                set_dataset_id(_buf_out, frame_id_out, dset_id_out);
                mark_frame_full(_buf_out, unique_name.c_str(), frame_id_out++);
            } else {
                dropped_frame_counter->labels({std::to_string(freq_id)}).inc();
            }

            mark_frame_empty(_buf_in_vis, unique_name.c_str(), frame_id_in_vis++);
            frame_counter->labels({std::to_string(freq_id)}).inc();
        }
        mark_frame_empty(_buf_in_sk, unique_name.c_str(), frame_id_in_sk++);
    }
}


// mostly copied from VisFrameView
void RfiFrameDrop::copy_frame(Buffer* buf_src, int frame_id_src, Buffer* buf_dest,
                              int frame_id_dest) {

    // Buffer sizes must match exactly
    if (buf_src->frame_size != buf_dest->frame_size) {
        throw std::runtime_error(
            fmt::format(fmt("Buffer sizes must match for direct copy (src {:d} != dest {:d})."),
                        buf_src->frame_size, buf_dest->frame_size));
    }

    int num_consumers = get_num_consumers(buf_src);

    // Copy or transfer the data part.
    if (num_consumers == 1) {
        // Transfer frame contents with directly...
        swap_frames(buf_src, frame_id_src, buf_dest, frame_id_dest);
    } else if (num_consumers > 1) {
        // Copy the frame data over, leaving the source intact
        std::memcpy(buf_dest->frames[frame_id_dest], buf_src->frames[frame_id_src],
                    buf_src->frame_size);
    }

    pass_metadata(buf_src, frame_id_src, buf_dest, frame_id_dest);
}

bool RfiFrameDrop::rest_enable_callback(nlohmann::json& update) {
    bool enable_rfi_zero_new;

    try {
        enable_rfi_zero_new = update.at("rfi_zeroing").get<bool>();
    } catch (nlohmann::json::exception& e) {
        WARN("Failure parsing update: Can't read 'rfi_zeroing' (bool): {:s}", e.what());
        return false;
    }

    if (enable_rfi_zero_new) {
        INFO("Enabled RFI frame dropping.");
    } else {
        INFO("Disabled RFI frame dropping.");
    }

    std::lock_guard<std::mutex> lock(update_mutex);

    // set enabled and state ID, but copy the rest from the last state
    if (state_ptr)
        std::tie(state_id, state_ptr) =
            dm.create_state<RFIFrameDropState>(enable_rfi_zero_new, state_ptr->get_thresholds());
    else {
        // If this callback is ran before the other one for the first time, state_ptr is not
        // initialized yet.
        std::vector<std::pair<float, float>> dummy_thresholds;
        std::tie(state_id, state_ptr) =
            dm.create_state<RFIFrameDropState>(enable_rfi_zero_new, dummy_thresholds);
    }


    return true;
}

bool RfiFrameDrop::rest_thresholds_callback(nlohmann::json& update) {
    nlohmann::json j;
    try {
        j = update.at("thresholds");
    } catch (nlohmann::json::exception& e) {
        WARN("Failure parsing update: array 'thresholds' not found: {:s}", e.what());
    }

    if (!j.is_array()) {
        WARN("Failure parsing update: entry 'thresholds' is not a list : {:s}", j.dump());
        return false;
    }

    auto thresholds_new = std::vector<std::pair<float, float>>();
    num_sk.clear();

    // Lock all update data, since num_sk is changed in this loop
    std::lock_guard<std::mutex> lock(update_mutex);
    for (const auto& t : j) {
        if (!t.is_object()) {
            WARN("Failure parsing update: item in list 'thresholds' is not a dict : {}", t.dump());
            return false;
        }

        float threshold, fraction;
        try {
            threshold = t["threshold"].get<float>();
        } catch (nlohmann::json::exception& e) {
            WARN("Failure parsing update: Required key `threshold` not present in item: {}",
                 t.dump());
            return false;
        }
        try {
            fraction = t["fraction"].get<float>();
        } catch (nlohmann::json::exception& e) {
            WARN("Failure parsing update: Required key `fraction` not present in item: {}",
                 t.dump());
            return false;
        }

        thresholds_new.push_back({threshold, fraction});
        num_sk.push_back((size_t)(fraction * sk_samples_per_frame));
    }


    // Resize sk_exceeds to length of thresholds
    sk_exceeds.resize(thresholds_new.size(), 0);

    INFO("Setting new RFI excision cuts:");
    for (const auto& [threshold, fraction] : thresholds_new)
        INFO("  added cut with threshold={}, fraction={}", threshold, fraction);

    // If this callback is ran before the other one for the first time, state_ptr is not
    // initialized yet.
    bool enabled = state_ptr ? state_ptr->get_enabled() : false;

    // build a new dataset state, copy enable-value
    std::tie(state_id, state_ptr) = dm.create_state<RFIFrameDropState>(enabled, thresholds_new);

    return true;
}
