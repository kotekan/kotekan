#include "receiveFlags.hpp"

#include "configUpdater.hpp"
#include "datasetManager.hpp"
#include "errors.h"
#include "prometheusMetrics.hpp"
#include "visBuffer.hpp"

#include <exception>
#include <utility>

using namespace std::placeholders;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::configUpdater;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(receiveFlags);


receiveFlags::receiveFlags(Config& config, const string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&receiveFlags::main_thread, this)),
    late_updates_counter(
        Metrics::instance().add_counter("kotekan_receiveflags_late_update_count", unique_name)) {
    // Setup the buffers
    buf_in = get_buffer("in_buf");
    buf_out = get_buffer("out_buf");
    register_consumer(buf_in, unique_name.c_str());
    register_producer(buf_out, unique_name.c_str());

    // Apply kotekan config
    int num = config.get<int>(unique_name, "num_elements");
    if (num < 0)
        throw std::invalid_argument("receiveFlags: config: invalid value for"
                                    " num_elements: "
                                    + std::to_string(num));
    num_elements = (size_t)num;
    num_kept_updates = config.get_default<uint32_t>(unique_name, "num_kept_updates", 5);

    /// FIFO for flags updates
    flags = updateQueue<std::pair<state_id_t, std::vector<float>>>(num_kept_updates);

    // we are ready to receive updates with the callback function now!
    // register as a subscriber with configUpdater
    configUpdater::instance().subscribe(this, std::bind(&receiveFlags::flags_callback, this, _1));
}

bool receiveFlags::flags_callback(nlohmann::json& json) {
    std::vector<float> flags_received(num_elements);
    std::fill(flags_received.begin(), flags_received.end(), 1.0);
    double ts;
    std::string update_id;

    // receive flags and start_time
    try {
        if (!json.at("bad_inputs").is_array())
            throw std::invalid_argument("receiveFlags: flags_callback "
                                        "received bad value 'bad_inputs': "
                                        + json.at("bad_inputs").dump());
        update_id = json.at("update_id").get<std::string>();

        if (!json.at("update_id").is_string())
            throw std::invalid_argument("receiveFlags: flags_callback "
                                        "received bad value 'update_id': "
                                        + json.at("update_id").dump());

        if (json.at("bad_inputs").size() > num_elements)
            throw std::invalid_argument(
                "receiveFlags: flags_callback "
                "received "
                + std::to_string(json.at("bad_inputs").size())
                + " bad inputs (has to be less than or equal num_elements = "
                + std::to_string(num_elements) + ").");
        if (!json.at("start_time").is_number())
            throw std::invalid_argument("receiveFlags: received bad value "
                                        "'start_time': "
                                        + json.at("start_time").dump());
        if (json.at("start_time") < 0)
            throw std::invalid_argument("receiveFlags: received negative "
                                        "start_time: "
                                        + json.at("start_time").dump());

        ts = json.at("start_time");

        for (nlohmann::json::iterator flag = json.at("bad_inputs").begin();
             flag != json.at("bad_inputs").end(); flag++) {
            if (*flag >= num_elements)
                throw std::invalid_argument("receiveFlags: received "
                                            "out-of-range bad_input: "
                                            + json.at("bad_inputs").dump());
            flags_received.at(*flag) = 0.0;
        }
    } catch (std::exception& e) {
        WARN("receiveFlags: Failure parsing message: {:s}", e.what());
        return false;
    }

    if (ts_frame > double_to_ts(ts)) {
        WARN("receiveFlags: Received update with a start_time that is older than the current "
             "frame (The difference is {:f} s).",
             ts_to_double(ts_frame) - ts);
        late_updates_counter.inc();
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

void receiveFlags::main_thread() {

    auto& dm = datasetManager::instance();

    uint32_t frame_id_in = 0;
    uint32_t frame_id_out = 0;

    timespec ts_late = {0, 0};

    std::pair<timespec, const std::vector<float>*> update;

    auto& receiveflags_update_age_metric =
        Metrics::instance().add_gauge("kotekan_receiveflags_update_age_seconds", unique_name);
    receiveflags_update_age_metric.set(-ts_to_double(ts_late));
    // Report number of frames received late
    auto& receiveflags_late_frame_counter =
        Metrics::instance().add_counter("kotekan_receiveflags_late_frame_count", unique_name);

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
        auto frame_out = visFrameView::copy_frame(buf_in, frame_id_in, buf_out, frame_id_out);

        // get the frames timestamp
        ts_frame = std::get<1>(frame_out.time);

        // Copy flags into frame
        flags_lock.lock();

        // Get and unpack the update
        const auto& [ts_update, update] = flags.get_update(ts_frame);

        if (update == nullptr) {
            FATAL_ERROR("receiveFlags: Flags for frame {:d} with timestamp {:f} are not in memory. "
                        "updateQueue is empty",
                        frame_id_in, ts_to_double(ts_frame));
            return;
        }

        const auto& [state_id, flag_list] = *update;

        std::pair<state_id_t, dset_id_t> key = {state_id, frame_out.dataset_id};
        if (output_dataset_ids.count(key) == 0) {
            double start = current_time();
            output_dataset_ids[key] = dm.add_dataset(state_id, frame_out.dataset_id);
            INFO("Adding flags to dM. Took {:.2f}s", current_time() - start);
        }
        frame_out.dataset_id = output_dataset_ids[key];

        ts_late = ts_update - ts_frame;
        if (ts_late.tv_sec > 0 && ts_late.tv_nsec > 0) {
            // This frame is too old,we don't have flags for it
            // --> Use the last update we have
            WARN("receiveFlags: Flags for frame {:d} with timestamp {:f} are not in memory. "
                 "Applying oldest flags found ({:f}). Consider increasing num_kept_updates.",
                 frame_id_in, ts_to_double(ts_frame), ts_to_double(ts_update));
            receiveflags_late_frame_counter.inc();
        }
        // actually copy the new flags and apply them from now
        std::copy(flag_list.begin(), flag_list.end(), frame_out.flags.begin());
        flags_lock.unlock();

        // Report how old the flags being applied to the current data are.
        receiveflags_update_age_metric.set(-ts_to_double(ts_late));

        // Mark output frame full and input frame empty
        mark_frame_full(buf_out, unique_name.c_str(), frame_id_out);
        mark_frame_empty(buf_in, unique_name.c_str(), frame_id_in);
        // Move forward one frame
        frame_id_out = (frame_id_out + 1) % buf_out->num_frames;
        frame_id_in = (frame_id_in + 1) % buf_in->num_frames;
    }
}
