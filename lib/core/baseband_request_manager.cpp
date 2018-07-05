#include "baseband_request_manager.hpp"

#include <iostream>
#include <sstream>


static void to_json(json& j, const BasebandRequest& r) {
    std::time_t received_c = std::chrono::system_clock::to_time_t(r.received - std::chrono::hours(24));
    std::stringstream received;
    received << std::put_time(std::localtime(&received_c), "%F %T");

    j = json{{"event_id", r.event_id},
             {"start", r.start_fpga},
             {"length", r.length_fpga},
             {"file_name", r.file_name},
             {"received", received.str()}};
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
  rest_server->register_post_callback("/baseband",
                                      std::bind(&BasebandRequestManager::handle_request_callback, this, _1, _2));
}

void BasebandRequestManager::status_callback(connectionInstance& conn){
    json requests_json = json::array();
    std::lock_guard<std::mutex> lock(requests_lock);

    for (auto& element : requests) {
        // XXX Compiler complains that this isn't used.
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
    json event_id_json = request["event_id"];
    json start_json = request["start"];
    json length_json = request["length"];
    json file_name_json = request["file_name"];
    json freq_id_json = request["freq_id"];
    if (event_id_json.is_number_integer() &&
        start_json.is_number_integer() &&
        length_json.is_number_integer() &&
        file_name_json.is_string() &&
        freq_id_json.is_number_integer()) {
        uint64_t event_id = request["event_id"];
        int64_t start_fpga = request["start"];
        int64_t length_fpga = request["length"];
        std::string file_name = request["file_name"];
        uint32_t freq_id = request["freq_id"];
        {
            std::lock_guard<std::mutex> lock(requests_lock);
            requests[freq_id].push_back({event_id, start_fpga, length_fpga, file_name, now});
        }
        requests_cv.notify_all();
        conn.send_empty_reply(HTTP_RESPONSE::OK);
    }
    else {
        conn.send_empty_reply(HTTP_RESPONSE::BAD_REQUEST);
    }
}


std::shared_ptr<BasebandDumpStatus> BasebandRequestManager::get_next_request(const uint32_t freq_id) {
    std::cout << "Waiting for notification\n";
    std::unique_lock<std::mutex> lock(requests_lock);

    using namespace std::chrono_literals;
    if (requests_cv.wait_for(lock, 0.1s) == std::cv_status::no_timeout) {
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
