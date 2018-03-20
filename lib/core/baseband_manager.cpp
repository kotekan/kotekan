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
}

void BasebandManager::status_callback(connectionInstance& conn){
    conn.send_empty_reply(501);
}
