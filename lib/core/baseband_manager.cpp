#include "baseband_manager.hpp"

BasebandManager::BasebandManager() {
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
    conn.send_empty_reply(501);
}

void BasebandManager::handle_request_callback(connectionInstance& conn, json& request){
  conn.send_empty_reply(502);
}
