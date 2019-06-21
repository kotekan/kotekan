#include "basebandApiManager.hpp"

#include "basebandReadoutManager.hpp"
#include "kotekanLogging.hpp"

#include <iostream>
#include <sstream>


namespace kotekan {

basebandReadoutManager&
    basebandApiManager::basebandReadoutRegistry::operator[](const uint32_t& key) {
    std::lock_guard<std::mutex> lock(map_lock);
    return readout_map[key];
}

basebandApiManager::basebandReadoutRegistry::iterator
basebandApiManager::basebandReadoutRegistry::begin() noexcept {
    std::lock_guard<std::mutex> lock(map_lock);
    return readout_map.begin();
}

basebandApiManager::basebandReadoutRegistry::iterator
basebandApiManager::basebandReadoutRegistry::end() noexcept {
    std::lock_guard<std::mutex> lock(map_lock);
    return readout_map.end();
}

void to_json(json& j, const basebandDumpStatus& d) {
    j = json{{"file_name", d.request.file_name},
             {"total", d.bytes_total},
             {"remaining", d.bytes_remaining}};

    switch (d.state) {
        case basebandDumpStatus::State::WAITING:
            j["status"] = "waiting";
            break;
        case basebandDumpStatus::State::INPROGRESS:
            j["status"] = "inprogress";
            break;
        case basebandDumpStatus::State::DONE:
            j["status"] = "done";
            break;
        case basebandDumpStatus::State::ERROR:
            j["status"] = "error";
            j["reason"] = d.reason;
            break;
        default:
            j["status"] = "error";
            j["reason"] = "Internal: Unknown status code";
    }
}

basebandApiManager::basebandApiManager() :
    // clang-format off
    request_count_metric(
        prometheusMetrics::instance()
        .add_stage_counter("kotekan_baseband_requests_total", "baseband")
        ) // clang-format on
{
}

basebandApiManager& basebandApiManager::instance() {
    static basebandApiManager _instance;
    return _instance;
}

void basebandApiManager::register_with_server(restServer* rest_server) {
    using namespace std::placeholders;
    rest_server->register_get_callback(
        "/baseband", std::bind(&basebandApiManager::status_callback_all, this, _1));
    rest_server->register_post_callback(
        "/baseband", std::bind(&basebandApiManager::handle_request_callback, this, _1, _2));
}

void basebandApiManager::status_callback_all(connectionInstance& conn) {
    std::map<std::string, std::vector<json>> event_readout_status;

    for (auto& element : readout_registry) {
        uint32_t freq_id = element.first;
        auto& readout_manager = element.second;

        for (auto event : readout_manager.all()) {
            json j(event);
            j["freq_id"] = freq_id;
            event_readout_status[std::to_string(event.request.event_id)].push_back(j);
        }
    }

    conn.send_json_reply(json(event_readout_status));
}


void basebandApiManager::status_callback_single_event(const uint64_t event_id,
                                                      connectionInstance& conn) {
    std::vector<json> event_status;

    for (auto& element : readout_registry) {
        uint32_t freq_id = element.first;
        auto& readout_manager = element.second;

        auto event = readout_manager.find(event_id);
        if (event) {
            json j(*event);
            j["freq_id"] = freq_id;
            event_status.push_back(j);
        }
    }

    if (event_status.empty()) {
        conn.send_empty_reply(HTTP_RESPONSE::NOT_FOUND);
    } else {
        conn.send_json_reply(json(event_status));
    }
}

basebandApiManager::basebandSlice
basebandApiManager::translate_trigger(const int64_t fpga_time0, const int64_t fpga_width,
                                      const double dm, const double dm_error,
                                      const uint32_t freq_id, const double ref_freq_hz) {
    const double freq = ADC_SAMPLE_RATE + FPGA_DELTA_FREQ * freq_id;
    const double freq_inv_sq_diff = (1. / (freq * freq) - 1. / (ref_freq_hz * ref_freq_hz));
    double min_delay = K_DM * (dm - N_DM_ERROR_TOL * dm_error) * freq_inv_sq_diff;
    double max_delay = K_DM * (dm + N_DM_ERROR_TOL * dm_error) * freq_inv_sq_diff;
    DEBUG("min DM delay: %lf, max DM delay, %lf", min_delay, max_delay);

    int64_t min_delay_fpga = round(min_delay * FPGA_FRAME_RATE);
    int64_t max_delay_fpga = round(max_delay * FPGA_FRAME_RATE);

    return {fpga_time0 >= 0 ? fpga_time0 + max_delay_fpga : fpga_time0,
            fpga_width + (min_delay_fpga - max_delay_fpga)};
}


void basebandApiManager::handle_request_callback(connectionInstance& conn, json& request) {
    auto now = std::chrono::system_clock::now();
    try {
        uint64_t event_id = request["event_id"];
        int64_t start_unix_seconds = request["start_unix_seconds"];

        int64_t start_fpga;
        if (start_unix_seconds >= 0) {
            int64_t start_unix_nano = request["start_unix_nano"];
            struct timespec ts = {start_unix_seconds, start_unix_nano};
            start_fpga = compute_fpga_seq(ts);
        } else {
            start_fpga = -1;
        }

        int64_t duration_nano = request["duration_nano"];
        int64_t duration_fpga = duration_nano / FPGA_PERIOD_NS;

        std::string file_path = request["file_path"];
        // Ensure there is no trailing slash
        while (file_path.back() == '/') {
            file_path.pop_back();
        }
        const double dm = request["dm"];
        const double dm_error = request["dm_error"];

        json response = json::object({});
        for (auto& element : readout_registry) {
            const uint32_t freq_id = element.first;
            auto& readout_entry = element.second;

            const std::string readout_file_name =
                "baseband_" + std::to_string(event_id) + "_" + std::to_string(freq_id) + ".h5";

            const auto readout_slice =
                translate_trigger(start_fpga, duration_fpga, dm, dm_error, freq_id);
            readout_entry.add({event_id, readout_slice.start_fpga, readout_slice.length_fpga,
                               file_path, readout_file_name, now});
            response[std::to_string(freq_id)] = json{{"file_name", readout_file_name},
                                                     {"start_fpga", readout_slice.start_fpga},
                                                     {"length_fpga", readout_slice.length_fpga}};
        }

        restServer& rest_server = restServer::instance();
        rest_server.register_get_callback("/baseband/" + std::to_string(event_id),
                                          [event_id, this](connectionInstance& nc) {
                                              status_callback_single_event(event_id, nc);
                                          });

        request_count_metric->inc();

        conn.send_json_reply(response);
    } catch (const std::exception& ex) {
        INFO("Bad baseband request: %s", ex.what());
        conn.send_empty_reply(HTTP_RESPONSE::BAD_REQUEST);
    }
}


basebandReadoutManager& basebandApiManager::register_readout_stage(const uint32_t freq_id) {
    return readout_registry[freq_id];
}

} // namespace kotekan
