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

void restServer::register_packet_callback(std::function<uint8_t*(int, int&) > callback, int port) {
    packet_callbacks[port] = callback;
}

void restServer::handle_packet_grab(mg_connection* nc, int ev, void* ev_data) {

    struct http_message *msg = (struct http_message *)ev_data;

    INFO("Handle packet grab message details: %s", msg->message.p);

    int port = -1;
    int num_packets = -1;

    try {
        json message = json::parse(string(msg->body.p, msg->body.len));
        port = message["port"];
        num_packets = message["num_packets"];
    } catch (...) {
        INFO("restServer: Parse failed handle_packet_grab, message %s");
        mg_send_head(nc, STATUS_BAD_REQUEST, 0, NULL);
        return;
    }

    if (num_packets < 0 || num_packets > 100) {
        INFO("restServer: handle_packet_grap: num_packets=%d out of range", num_packets);
        mg_send_head(nc, STATUS_BAD_REQUEST, 0, NULL);
        return;
    }
    if (port < 0 || port > 8) {
        INFO("restServer: handle_packet_grap: port=%d out of range", port);
        mg_send_head(nc, STATUS_BAD_REQUEST, 0, NULL);
        return;
    }

    int len;
    uint8_t * packets = __rest_server->packet_callbacks[port](num_packets, len);

    if (packets != nullptr) {
        mg_send_head(nc, STATUS_OK, len, "Content-Type: application/octet-stream");
        mg_send(nc, (void *)packets, len);
    } else {
        mg_send_head(nc, STATUS_REQUEST_FAILED, 0, NULL);
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
    mg_register_http_endpoint(nc, "/packet_grab", handle_packet_grab);

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

