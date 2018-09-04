#include "basebandRequestManager.hpp"

#include <iostream>
#include <sstream>

#include "kotekanLogging.hpp"

basebandRequestManager::basebandReadoutRegistryEntry& basebandRequestManager::basebandReadoutRegistry::operator[]( const uint32_t& key ) {
    std::lock_guard<std::mutex> lock(map_lock);
    return readout_map[key];
}

basebandRequestManager::basebandReadoutRegistry::iterator basebandRequestManager::basebandReadoutRegistry::begin() noexcept {
    std::lock_guard<std::mutex> lock(map_lock);
    return readout_map.begin();
}

basebandRequestManager::basebandReadoutRegistry::iterator basebandRequestManager::basebandReadoutRegistry::end() noexcept {
    std::lock_guard<std::mutex> lock(map_lock);
    return readout_map.end();
}

void to_json(json &j, const basebandRequest& r) {
    std::time_t received_c = std::chrono::system_clock::to_time_t(r.received - std::chrono::hours(24));
    std::stringstream received;
    received << std::put_time(std::localtime(&received_c), "%F %T");

    j = json{{"event_id", r.event_id,},
             {"start", r.start_fpga,},
             {"length", r.length_fpga,},
             {"file_path", r.file_path,},
             {"file_name", r.file_name,},
             {"received", received.str()}};
}

void to_json(json &j, const basebandDumpStatus& d) {
    j = json{{"file_name", d.request.file_name},
             {"total", d.bytes_total},
             {"remaining", d.bytes_remaining}};

    switch(d.state) {
    case basebandDumpStatus::State::WAITING:
        j["status"] = "waiting"; break;
    case basebandDumpStatus::State::INPROGRESS:
        j["status"] = "inprogress"; break;
    case basebandDumpStatus::State::DONE:
        j["status"] = "done"; break;
    case basebandDumpStatus::State::ERROR:
        j["status"] = "error";
        j["reason"] = d.reason;
        break;
    default:
        j["status"] = "error";
        j["reason"] = "Internal: Unknown status code";
    }
}

basebandRequestManager& basebandRequestManager::instance() {
    static basebandRequestManager _instance;
    return _instance;
}

void basebandRequestManager::register_with_server(restServer* rest_server) {
  using namespace std::placeholders;
  rest_server->register_get_callback("/baseband",
                                     std::bind(&basebandRequestManager::status_callback_all, this, _1));
  rest_server->register_post_callback("/baseband",
                                      std::bind(&basebandRequestManager::handle_request_callback, this, _1, _2));
}

void basebandRequestManager::status_callback_all(connectionInstance& conn){
    std::map<std::string, std::vector<json>>  event_readout_status;

    for (auto& element : readout_registry) {
        uint32_t freq_id = element.first;
        auto& readout_entry = element.second;
        std::lock_guard<std::mutex> lock(readout_entry.requests_lock);
        for (auto& req : readout_entry.request_queue) {
            json j(req);
            j["freq_id"] = freq_id;
            event_readout_status[std::to_string(req.event_id)].push_back(j);
        }
        for (const auto& d : readout_entry.processing) {
            json j(d);
            j["freq_id"] = freq_id;
            event_readout_status[std::to_string(d.request.event_id)].push_back(j);
        }
        {
            std::lock_guard<std::mutex> lock(*readout_entry.current_lock);
            if (readout_entry.current_status) {
                json j(*readout_entry.current_status);
                j["freq_id"] = freq_id;
                event_readout_status[std::to_string(readout_entry.current_status->request.event_id)].push_back(j);
            }
        }
    }

    conn.send_json_reply(json(event_readout_status));
}


void basebandRequestManager::status_callback_single_event(const uint64_t event_id, connectionInstance& conn){
    std::vector<json> event_status;

    for (auto& element : readout_registry) {
        uint32_t freq_id = element.first;
        auto& readout_entry = element.second;

        std::lock_guard<std::mutex> lock(readout_entry.requests_lock);

        bool found = false;
        for (auto& req : readout_entry.request_queue) {
            if (req.event_id == event_id) {
                json j(req);
                j["freq_id"] = freq_id;
                event_status.push_back(j);
                found = true;
                break;
            }
        }
        if (found) {
            continue;
        }

        for (const auto& d : readout_entry.processing) {
            if (d.request.event_id == event_id) {
                json j(d);
                j["freq_id"] = freq_id;
                event_status.push_back(j);
                found = true;
                break;
            }
        }
        if (!found) {
            std::lock_guard<std::mutex> lock(*readout_entry.current_lock);
            if (readout_entry.current_status && readout_entry.current_status->request.event_id == event_id) {
                json j(*readout_entry.current_status);
                j["freq_id"] = freq_id;
                event_status.push_back(j);
            }
        }
    }

    if (event_status.empty()) {
        conn.send_empty_reply(HTTP_RESPONSE::NOT_FOUND);
    } else {
        conn.send_json_reply(json(event_status));
    }
}

basebandSlice basebandRequestManager::translate_trigger(const int64_t fpga_time0, const int64_t fpga_width,
                                                        const double dm, const double dm_error,
                                                        const int64_t freq_id,
                                                        const double ref_freq_hz) {
    const double freq = FPGA_FREQ0 + FPGA_DELTA_FREQ * freq_id;
    const double freq_inv_sq_diff = (1. / (freq*freq) - 1. / (ref_freq_hz*ref_freq_hz));
    double min_delay = K_DM * (dm - N_DM_ERROR_TOL * dm_error) * freq_inv_sq_diff;
    double max_delay = K_DM * (dm + N_DM_ERROR_TOL * dm_error) * freq_inv_sq_diff;
    DEBUG("min DM delay: %lf, max DM delay, %lf", min_delay, max_delay);

    int64_t min_delay_fpga = round(min_delay * FPGA_FRAME_RATE);
    int64_t max_delay_fpga = round(max_delay * FPGA_FRAME_RATE);

    return {fpga_time0 >= 0 ? fpga_time0 + max_delay_fpga : fpga_time0,
            fpga_width + (min_delay_fpga - max_delay_fpga)};
}


void basebandRequestManager::handle_request_callback(connectionInstance& conn, json& request){
    auto now = std::chrono::system_clock::now();
    try {
        uint64_t event_id = request["event_id"];
        int64_t start_unix_seconds = request["start_unix_seconds"];

        int64_t start_fpga;
        if (start_unix_seconds >= 0) {
            int64_t start_unix_nano = request["start_unix_nano"];
            struct timespec ts = {start_unix_seconds, start_unix_nano};
            start_fpga = compute_fpga_seq(ts);
        }
        else {
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

            std::lock_guard<std::mutex> lock(readout_entry.requests_lock);
            const std::string readout_file_name = "baseband_" +
                std::to_string(event_id) +
                "_" + std::to_string(freq_id) + ".h5";

            const auto readout_slice = translate_trigger(start_fpga,
                                                         duration_fpga,
                                                         dm, dm_error,
                                                         freq_id);
            readout_entry.request_queue.push_back({
                    event_id,
                    readout_slice.start_fpga,
                    readout_slice.length_fpga,
                    file_path,
                    readout_file_name,
                    now
            });
            readout_entry.requests_cv.notify_all();
            response[std::to_string(freq_id)] = json{
                {"file_name", readout_file_name},
                {"start_fpga", readout_slice.start_fpga},
                {"length_fpga", readout_slice.length_fpga}
            };
        }
        restServer &rest_server = restServer::instance();
        rest_server.register_get_callback("/baseband/" + std::to_string(event_id),
                                          [event_id, this](connectionInstance &nc) {
                                              status_callback_single_event(event_id, nc);
                                          });

        conn.send_json_reply(response);
    } catch (const std::exception &ex) {
        INFO("Bad baseband request: %s", ex.what());
        conn.send_empty_reply(HTTP_RESPONSE::BAD_REQUEST);
    }
}


std::shared_ptr<std::mutex> basebandRequestManager::register_readout_process(const uint32_t freq_id) {
    return readout_registry[freq_id].current_lock;
}

std::shared_ptr<basebandDumpStatus> basebandRequestManager::get_next_request(const uint32_t freq_id) {
    DEBUG("Waiting for notification");

    auto& readout_entry = readout_registry[freq_id];
    std::unique_lock<std::mutex> lock(readout_entry.requests_lock);

    // NB: the requests_lock is released while the thread is waiting on requests_cv, and reacquired once woken
    using namespace std::chrono_literals;
    if (readout_entry.requests_cv.wait_for(lock, 0.1s) == std::cv_status::no_timeout) {
        DEBUG("Notified");
    }
    else {
        DEBUG("Expired");
    }

    std::lock_guard<std::mutex> current_lock(*readout_entry.current_lock);
    if (readout_entry.current_status) {
        // if this method is called, we know that the readout is done with the
        // current dump
        readout_entry.processing.push_back(*readout_entry.current_status);
    }

    if (!readout_entry.request_queue.empty()) {
        basebandRequest req = readout_entry.request_queue.front();
        readout_entry.request_queue.pop_front();

        basebandDumpStatus s{req};
        readout_entry.current_status = std::make_shared<basebandDumpStatus>(s);
    }
    else {
        readout_entry.current_status = nullptr;
    }
    return readout_entry.current_status;
}
