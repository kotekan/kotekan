#include "ReceiveFlags.hpp"

#include "Config.hpp"            // for Config
#include "Hash.hpp"              // for operator<
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"              // for mark_frame_empty, mark_frame_full, register_consumer
#include "bufferContainer.hpp"   // for bufferContainer
#include "configUpdater.hpp"     // for configUpdater
#include "datasetManager.hpp"    // for dset_id_t, datasetManager, state_id_t
#include "datasetState.hpp"      // for flagState
#include "kotekanLogging.hpp"    // for WARN, INFO
#include "prometheusMetrics.hpp" // for Metrics, Counter, Gauge
#include "visBuffer.hpp"         // for VisFrameView
#include "visUtil.hpp"           // for frameID, ts_to_double, current_time, double_to_ts, modulo

#include "gsl-lite.hpp" // for span<>::iterator, span

#include <algorithm>   // for copy, fill, copy_backward, max
#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, _Placeholder, bind, _1, function
#include <memory>      // for operator==, __shared_ptr_access
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for invalid_argument, runtime_error
#include <tuple>       // for get
#include <type_traits> // for add_const<>::type
#include <utility>     // for pair, move, tuple_element<>::type

using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(ReceiveFlags);


ReceiveFlags::ReceiveFlags(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&ReceiveFlags::main_thread, this)),
    receiveflags_late_frame_counter(
        Metrics::instance().add_counter("kotekan_receiveflags_late_frame_count", unique_name, {})),
    receiveflags_update_age_metric(
        Metrics::instance().add_gauge("kotekan_receiveflags_update_age_seconds", unique_name, {})),
    late_updates_counter(
        Metrics::instance().add_counter("kotekan_receiveflags_late_update_count", unique_name, {})) {
    // Setup the buffers
    buf_in = get_buffer("in_buf");
    buf_out = get_buffer("out_buf");
    register_consumer(buf_in, unique_name.c_str());
    register_producer(buf_out, unique_name.c_str());

    // Apply kotekan config
    int num = config.get<int>(unique_name, "num_elements");
    if (num < 0)
        throw std::invalid_argument("ReceiveFlags: config: invalid value for"
                                    " num_elements: "
                                    + std::to_string(num));
    num_elements = static_cast<size_t>(num);
    num_kept_updates = config.get_default<uint32_t>(unique_name, "num_kept_updates", 5);

    /// FIFO for flags updates
    flags.resize(num_kept_updates);

    // we are ready to receive updates with the callback function now!
    // register as a subscriber with configUpdater
    configUpdater::instance().subscribe(this, std::bind(&ReceiveFlags::flags_callback, this, _1));
}

bool ReceiveFlags::flags_callback(nlohmann::json& json) {
    std::vector<float> flags_received(num_elements);
    std::fill(flags_received.begin(), flags_received.end(), 1.0);
    double ts;
    std::string update_id;

    // receive flags and start_time
    try {
        if (!json.at("bad_inputs").is_array())
            throw std::invalid_argument("ReceiveFlags: flags_callback "
                                        "received bad value 'bad_inputs': "
                                        + json.at("bad_inputs").dump());
        update_id = json.at("update_id").get<std::string>();

        if (!json.at("update_id").is_string())
            throw std::invalid_argument("ReceiveFlags: flags_callback "
                                        "received bad value 'update_id': "
                                        + json.at("update_id").dump());

        if (json.at("bad_inputs").size() > num_elements)
            throw std::invalid_argument(
                "ReceiveFlags: flags_callback received "
                + std::to_string(json.at("bad_inputs").size())
                + " bad inputs (has to be less than or equal num_elements = "
                + std::to_string(num_elements) + ").");
        if (!json.at("start_time").is_number())
            throw std::invalid_argument("ReceiveFlags: received bad value "
                                        "'start_time': "
                                        + json.at("start_time").dump());
        if (json.at("start_time") < 0)
            throw std::invalid_argument("ReceiveFlags: received negative "
                                        "start_time: "
                                        + json.at("start_time").dump());

        ts = json.at("start_time");

        for (nlohmann::json::iterator flag = json.at("bad_inputs").begin();
             flag != json.at("bad_inputs").end(); ++flag) {
            if (*flag >= num_elements)
                throw std::invalid_argument("ReceiveFlags: received "
                                            "out-of-range bad_input: "
                                            + json.at("bad_inputs").dump());
            flags_received.at(*flag) = 0.0;
        }
    } catch (std::exception& e) {
        WARN("Failure parsing message: {:s}", e.what());
        return false;
    }

    if (ts_frame > double_to_ts(ts)) {
        WARN("Received update with a start_time that is older than the current "
             "frame (The difference is {:f} s).",
             ts_to_double(ts_frame) - ts);
        late_updates_counter->labels({}).inc();
    }

    auto& dm = datasetManager::instance();
    auto state_id = dm.create_state<flagState>(update_id).first;

    // update the flags
    flags_lock.lock();
    flags.insert(double_to_ts(ts), {state_id, std::move(flags_received)});
    flags_lock.unlock();

    INFO("Updated flags to {:s}.", update_id);
    return true;
}

void ReceiveFlags::main_thread() {

    frameID frame_id_in(buf_in);
    frameID frame_id_out(buf_out);

    timespec ts_late = {0, 0};

    receiveflags_update_age_metric->labels({}).set(-ts_to_double(ts_late));

    while (!stop_thread) {

        // Wait for an input frame
        if (wait_for_full_frame(buf_in, unique_name.c_str(), frame_id_in) == nullptr) {
            break;
        }
        // wait for an empty output frame
        if (wait_for_empty_frame(buf_out, unique_name.c_str(), frame_id_out) == nullptr) {
            break;
        }

        // Copy frame into output buffer
        auto frame_out = VisFrameView::copy_frame(buf_in, frame_id_in, buf_out, frame_id_out);

        // get the frames timestamp
        ts_frame = std::get<1>(frame_out.time);

        // Output frame is only valid if we have a valid update for this frame
        bool success = copy_flags_into_frame(frame_out);

        // Mark input frame empty
        mark_frame_empty(buf_in, unique_name.c_str(), frame_id_in++);

        // Mark output frame full if we had valid flags. Otherwise drop it.
        if (success)
            mark_frame_full(buf_out, unique_name.c_str(), frame_id_out++);
    }
}

bool ReceiveFlags::copy_flags_into_frame(const VisFrameView& frame_out) {
    auto& dm = datasetManager::instance();

    std::lock_guard<std::mutex> lock(flags_lock);

    // Get and unpack the update
    const auto& [ts_update, update] = flags.get_update(ts_frame);

    // Report how old the flags being applied to the current data are.
    receiveflags_update_age_metric->labels({}).set(-ts_to_double(ts_update - ts_frame));

    if (update == nullptr) {
        WARN("updateQueue: {}\nFlags for frame with timestamp {:f} are not in memory. Dropping "
             "frame...",
             flags, ts_to_double(ts_frame));
        receiveflags_late_frame_counter->labels({}).inc();
        return false;
    }

    const auto& [state_id, flag_list] = *update;

    std::pair<state_id_t, dset_id_t> key = {state_id, frame_out.dataset_id};
    if (output_dataset_ids.count(key) == 0) {
        double start = current_time();
        output_dataset_ids[key] = dm.add_dataset(state_id, frame_out.dataset_id);
        INFO("Adding flags to dM. Took {:.2f}s", current_time() - start);
    }
    frame_out.dataset_id = output_dataset_ids[key];

    // actually copy the new flags and apply them from now
    std::copy(flag_list.begin(), flag_list.end(), frame_out.flags.begin());

    return true;
}
