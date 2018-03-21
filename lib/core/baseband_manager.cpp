#include "baseband_manager.hpp"

#include <iostream>
#include <sstream>


static void to_json(json& j, const BasebandRequest& r) {
    std::time_t received_c = std::chrono::system_clock::to_time_t(r.received - std::chrono::hours(24));
    std::stringstream received;
    received << std::put_time(std::localtime(&received_c), "%F %T");

    j = json{{"start", r.start_fpga}, {"length", r.length_fpga}, {"received", received.str()}};
}


BasebandManager& BasebandManager::instance() {
    static BasebandManager _instance;
    return _instance;
}

void BasebandManager::register_with_server(restServer* rest_server) {
  using namespace std::placeholders;
  rest_server->register_get_callback("/baseband",
                                     std::bind(&BasebandManager::status_callback, this, _1));
  rest_server->register_json_callback("/baseband",
                                      std::bind(&BasebandManager::handle_request_callback, this, _1, _2));
}

void BasebandManager::status_callback(connectionInstance& conn){
    json requests_json = json::array();

    {
        std::lock_guard<std::mutex> lock(requests_lock);
        for (auto& req : requests) {
            json j;
            to_json(j, req);
            requests_json.push_back(j);
        }
    }
    conn.send_text_reply(requests_json.dump());
}

void BasebandManager::handle_request_callback(connectionInstance& conn, json& request){
    auto now = std::chrono::system_clock::now();
    std::lock_guard<std::mutex> lock(requests_lock);
    json start_json = request["start"];
    json length_json = request["length"];
    if (start_json.is_number_integer() && length_json.is_number_integer()) {
        int64_t start_fpga = request["start"];
        int64_t length_fpga = request["length"];
        requests.push_back({start_fpga, length_fpga, now});
        conn.send_empty_reply(STATUS_OK);
    }
    else {
        conn.send_empty_reply(STATUS_BAD_REQUEST);
    }
}
