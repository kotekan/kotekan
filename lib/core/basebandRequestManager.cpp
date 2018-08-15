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

static json to_json(uint32_t freq_id, const basebandRequest& r) {
    std::time_t received_c = std::chrono::system_clock::to_time_t(r.received - std::chrono::hours(24));
    std::stringstream received;
    received << std::put_time(std::localtime(&received_c), "%F %T");

    json j = json{{"freq_id", freq_id},
             {"event_id", r.event_id},
             {"start", r.start_fpga},
             {"length", r.length_fpga},
             {"file_name", r.file_name},
             {"received", received.str()}};
    return j;
}

static json to_json(const basebandDumpStatus& d) {
    json j = json{{"total", d.bytes_total}, {"remaining", d.bytes_remaining}};
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
    return j;
}

basebandRequestManager& basebandRequestManager::instance() {
    static basebandRequestManager _instance;
    return _instance;
}

void basebandRequestManager::register_with_server(restServer* rest_server) {
  using namespace std::placeholders;
  rest_server->register_get_callback("/baseband",
                                     std::bind(&basebandRequestManager::status_callback, this, _1));
  rest_server->register_post_callback("/baseband",
                                      std::bind(&basebandRequestManager::handle_request_callback, this, _1, _2));
}

void basebandRequestManager::status_callback(connectionInstance& conn){
    json requests_json = json::array();

    for (auto& element : readout_registry) {
        uint32_t freq_id = element.first;
        auto& readout_entry = element.second;
        std::lock_guard<std::mutex> lock(readout_entry.requests_lock);
        for (auto& req : readout_entry.request_queue) {
            json j = to_json(freq_id, req);
            requests_json.push_back(j);
        }
        for (const auto& d : readout_entry.processing) {
            json j = to_json(d);
            requests_json.push_back(j);
        }
        {
            std::lock_guard<std::mutex> lock(*readout_entry.current_lock);
            if (readout_entry.current_status) {
                json j = to_json(*readout_entry.current_status);
                requests_json.push_back(j);
            }
        }
    }

    conn.send_text_reply(requests_json.dump());
}

void basebandRequestManager::handle_request_callback(connectionInstance& conn, json& request){
    auto now = std::chrono::system_clock::now();
    try {
        uint64_t event_id = request["event_id"];
        int64_t start_fpga = request["start"];
        int64_t length_fpga = request["length"];
        std::string file_name = request["file_name"];
        uint32_t freq_id = request["freq_id"];
        auto& readout_entry = readout_registry[freq_id];
        {
            std::lock_guard<std::mutex> lock(readout_entry.requests_lock);
            readout_entry.request_queue.push_back({event_id, start_fpga, length_fpga, file_name, now});
        }
        readout_entry.requests_cv.notify_all();
        conn.send_empty_reply(HTTP_RESPONSE::OK);
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

        readout_entry.current_status = std::make_shared<basebandDumpStatus>(basebandDumpStatus{req});
    }
    else {
        readout_entry.current_status = nullptr;
    }
    return readout_entry.current_status;
}
