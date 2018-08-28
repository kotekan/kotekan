#include "receiveFlags.hpp"
#include "errors.h"
#include "configUpdater.hpp"
#include "visBuffer.hpp"
#include "prometheusMetrics.hpp"

#include <exception>

using namespace std::placeholders;

REGISTER_KOTEKAN_PROCESS(receiveFlags);


receiveFlags::receiveFlags(Config &config, const string& unique_name,
        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
            std::bind(&receiveFlags::main_thread, this)) {
         // Setup the buffers
         buf_in = get_buffer("in_buf");
         buf_out = get_buffer("out_buf");
         register_consumer(buf_in, unique_name.c_str());
         register_producer(buf_out, unique_name.c_str());

         //Apply kotekan config
         apply_config(0);

         /// FIFO for flags updates
         flags = updateQueue<std::vector<float>>(num_kept_updates);

         // we are ready to receive updates with the callback function now!
         // register as a subscriber with configUpdater
         configUpdater::instance().subscribe(this,
                            std::bind(&receiveFlags::flags_callback, this, _1));
}

void receiveFlags::apply_config(uint64_t fpga_seq) {
    (void)fpga_seq;
    int num = config.get_int(unique_name, "num_elements");
    if (num < 0)
        throw std::invalid_argument("receiveFlags: config: invalid value for" \
                                    " num_elements: "
                                    + std::to_string(num));
    num_elements = (size_t)num;

    updatable_config = config.get_string(unique_name, "updatable_config");

    num_kept_updates = config.get_uint32_default(unique_name,
                                                 "num_kept_updates", 5);
}

bool receiveFlags::flags_callback(nlohmann::json &json) {
    std::vector<float> flags_received (num_elements);
    std::fill(flags_received.begin(), flags_received.end(), 1.0);
    double ts;

    //receive flags and timestamp
    try {
        if (!json.at("bad_inputs").is_array())
            throw std::invalid_argument("receiveFlags: flags_callback " \
                                        "received bad value 'bad_inputs': " +
                                        json.at("bad_inputs").dump());
        if (json.at("bad_inputs").size() > num_elements)
            throw std::invalid_argument("receiveFlags: flags_callback " \
                      "received " + std::to_string(json.at("bad_inputs").size())
                      + " bad inputs (has to be less than or equal num_elements = "
                      + std::to_string(num_elements) + ").");
        if (!json.at("timestamp").is_number())
            throw std::invalid_argument("receiveFlags: received bad value " \
                                        "'timestamp': " +
                                        json.at("timestamp").dump());
        if (json.at("timestamp") < 0)
            throw std::invalid_argument("receiveFlags: received negative " \
                                       "timestamp: " +
                                       json.at("timestamp").dump());

        ts = json.at("timestamp");

        for (nlohmann::json::iterator flag = json.at("bad_inputs").begin();
             flag != json.at("bad_inputs").end(); flag++) {
            if (*flag >= num_elements)
                throw std::invalid_argument("receiveFlags: received " \
                                            "out-of-range bad_input: " +
                                            json.at("bad_inputs").dump());
            flags_received.at(*flag) = 0.0;
        }
    } catch (std::exception& e) {
        WARN("receiveFlags: Failure parsing message: %s", e.what());
        return false;
    }

    if (ts_frame > double_to_ts(ts)) {
        WARN("receiveFlags: Received update with a timestamp that is older " \
             "than the current frame (The difference is %f s).",
             ts_to_double(ts_frame) - ts);
        prometheusMetrics::instance().add_process_metric(
            "kotekan_receiveflags_old_update_seconds",
            unique_name, ts_to_double(ts_frame) - ts);
    }

    // update the flags
    flags_lock.lock();
    flags.insert(double_to_ts(ts), std::move(flags_received));
    flags_lock.unlock();

    INFO("Updated flags to %s.", json.at("tag").get<std::string>().c_str());

    return true;
}

void receiveFlags::main_thread() {

    uint32_t frame_id_in = 0;
    uint32_t frame_id_out = 0;

    timespec ts_late = {0,0};

    std::pair<timespec, const std::vector<float>*> update;

    while (!stop_thread) {
        // Wait for an input frame
        if(wait_for_full_frame(buf_in, unique_name.c_str(),
                               frame_id_in) == nullptr) {
            break;
        }
        //wait for an empty output frame
        if(wait_for_empty_frame(buf_out, unique_name.c_str(),
                                frame_id_out) == nullptr) {
            break;
        }

        // Copy frame into output buffer
        auto frame_out = visFrameView::copy_frame(buf_in, frame_id_in,
                                                  buf_out, frame_id_out);

        // get the frames timestamp
        ts_frame = std::get<1>(frame_out.time);

        // Copy flags into frame
        flags_lock.lock();
        update = flags.get_update(ts_frame);
        if (update.second == nullptr) {
            ERROR("receiveFlags: Flags for frame %d with timestamp %f are" \
                  "not in memory. updateQueue is empty",
                  frame_id_in, ts_to_double(ts_frame));
        }
        ts_late = update.first - ts_frame;
        if (ts_late.tv_sec > 0 && ts_late.tv_nsec > 0) {
            // This frame is too old,we don't have flags for it
            // --> Use the last update we have
            WARN("receiveFlags: Flags for frame %d with timestamp %f are" \
                  "not in memory. Applying oldest flags found. (%d)" \
                 " Concider increasing num_kept_updates.", frame_id_in,
                 ts_to_double(ts_frame), ts_to_double(update.first));
            prometheusMetrics::instance().add_process_metric(
                "kotekan_receiveflags_old_frame_seconds",
                unique_name, ts_to_double(ts_late));
        }
        // actually copy the new flags and apply them from now
        std::copy(update.second->begin(), update.second->end(),
                  frame_out.flags.begin());
        flags_lock.unlock();

        // Report how old the flags being applied to the current data are.
        prometheusMetrics::instance().add_process_metric(
            "kotekan_receiveflags_update_age_seconds",
            unique_name, -ts_to_double(ts_late));

        // Mark output frame full and input frame empty
        mark_frame_full(buf_out, unique_name.c_str(), frame_id_out);
        mark_frame_empty(buf_in, unique_name.c_str(), frame_id_in);
        // Move forward one frame
        frame_id_out = (frame_id_out + 1) % buf_out->num_frames;
        frame_id_in = (frame_id_in + 1) % buf_in->num_frames;
    }
}

