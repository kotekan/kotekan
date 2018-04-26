#include "baseband_request_manager.hpp"

#include <iostream>
#include <sstream>


static void to_json(json& j, const BasebandRequest& r) {
    std::time_t received_c = std::chrono::system_clock::to_time_t(r.received - std::chrono::hours(24));
    std::stringstream received;
    received << std::put_time(std::localtime(&received_c), "%F %T");

    j = json{{"start", r.start_fpga}, {"length", r.length_fpga}, {"received", received.str()}};
}

static void to_json(json& j, const BasebandDumpStatus& d) {
    j = json{{"total", d.bytes_total}, {"remaining", d.bytes_remaining}};
}

BasebandRequestManager& BasebandRequestManager::instance() {
    static BasebandRequestManager _instance;
    return _instance;
}

void BasebandRequestManager::register_with_server(restServer* rest_server) {
  using namespace std::placeholders;
  rest_server->register_get_callback("/baseband",
                                     std::bind(&BasebandRequestManager::status_callback, this, _1));
  rest_server->register_json_callback("/baseband",
                                      std::bind(&BasebandRequestManager::handle_request_callback, this, _1, _2));
}

void BasebandRequestManager::status_callback(connectionInstance& conn){
    json requests_json = json::array();
    std::lock_guard<std::mutex> lock(requests_lock);

    for (auto& element : requests) {
        uint32_t freq_id = element.first;
        for (auto& req : element.second) {
            json j;
            to_json(j, req);
            requests_json.push_back(j);
        }
    }

    for (const auto& d : processing) {
        json j;
        to_json(j, *d);
        requests_json.push_back(j);
    }

    conn.send_text_reply(requests_json.dump());
}

void BasebandRequestManager::handle_request_callback(connectionInstance& conn, json& request){
    auto now = std::chrono::system_clock::now();
    json start_json = request["start"];
    json length_json = request["length"];
    json freq_id_json = request["freq_id"];
    if (start_json.is_number_integer() &&
        length_json.is_number_integer() &&
        freq_id_json.is_number_integer()) {
        int64_t start_fpga = request["start"];
        int64_t length_fpga = request["length"];
        uint32_t freq_id = request["freq_id"];
        {
            std::lock_guard<std::mutex> lock(requests_lock);
            requests[freq_id].push_back({start_fpga, length_fpga, now});
        }
        requests_cv.notify_all();
        conn.send_empty_reply(STATUS_OK);
    }
    else {
        conn.send_empty_reply(STATUS_BAD_REQUEST);
    }
}


std::shared_ptr<BasebandDumpStatus> BasebandRequestManager::get_next_request(const uint32_t freq_id) {
    std::cout << "Waiting for notification\n";
    std::unique_lock<std::mutex> lock(requests_lock);

    using namespace std::chrono_literals;
    if (requests_cv.wait_for(lock, 1s) == std::cv_status::no_timeout) {
        std::cout << "Notified\n";
    }
    else {
        std::cout << "Expired\n";
    }

    if (!requests[freq_id].empty()) {
        BasebandRequest req = requests[freq_id].front();
        auto task = std::make_shared<BasebandDumpStatus>(BasebandDumpStatus{req});
        requests[freq_id].pop_front();
        processing.push_back(task);
        return task;
    }
    else {
        return nullptr;
    }
}
