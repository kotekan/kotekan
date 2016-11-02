#include "restServer.hpp"
#include "errors.h"

#include <pthread.h>
#include <sched.h>
#include <string>

using json = nlohmann::json;
using std::string;

#define STATUS_OK 200
#define STATUS_BAD_REQUEST 400
#define STATUS_REQUEST_FAILED 402
#define STATUS_NOT_FOUND 404
#define STATUS_INTERNAL_ERROR 500

restServer * __rest_server = new restServer();

restServer * get_rest_server() {
    return __rest_server;
}

restServer::restServer() : main_thread() {
}

restServer::~restServer() {
}

void restServer::handle_status(mg_connection* nc, int ev, void* ev_data) {
    struct http_message *msg = (struct http_message *)ev_data;
    INFO("Message details: uri %s", msg->message.p);

    json status = {{"status", "ok"}};
    string status_str = status.dump(0);
    int len = status_str.length();

    mg_send_head(nc, STATUS_OK, len, "Content-Type: application/json");
    mg_send(nc, status_str.c_str(), len);
}

void restServer::handle_notfound(mg_connection* nc, int ev, void* ev_data) {
    switch (ev) {
        case MG_EV_HTTP_REQUEST:
            mg_send_head(nc, STATUS_NOT_FOUND, 0, NULL);
            break;
    }
}

void restServer::mongoose_thread() {

    // init http server
    mg_mgr_init(&mgr, NULL);
    nc = mg_bind(&mgr, port, handle_notfound);
    if (!nc) {
        INFO("restServer: cannot bind to %s", port);
        return;
    }
    mg_set_protocol_http_websocket(nc);

    // add endpoints
    mg_register_http_endpoint(nc, "/status", handle_status);

    INFO("restServer: started server on port %s", port);

    // run event loop
    for (;;) mg_mgr_poll(&mgr, 1000);

    mg_mgr_free(&mgr);
}

void restServer::start() {
    main_thread = std::thread(&restServer::mongoose_thread, this);

    // Requires Linux, this could possibly be made more general someday.
    // TODO Move to config
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int j = 4; j < 12; j++)
        CPU_SET(j, &cpuset);
    pthread_setaffinity_np(main_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
}

