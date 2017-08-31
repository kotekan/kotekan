#include "restServer.hpp"
#include "errors.h"

#include <pthread.h>
#include <sched.h>
#include <string>

using json = nlohmann::json;
using std::string;

restServer * __rest_server = new restServer();

restServer * get_rest_server() {
    return __rest_server;
}

restServer::restServer() : main_thread() {
}

restServer::~restServer() {
}

void restServer::handle_request(mg_connection* nc, int ev, void* ev_data) {
    if (ev != MG_EV_HTTP_REQUEST)
        return;

    struct http_message *msg = (struct http_message *)ev_data;
    string url = string(msg->uri.p, msg->uri.len);

    if (!__rest_server->json_callbacks.count(url)) {
        mg_send_head(nc, STATUS_NOT_FOUND, 0, NULL);
        return;
    }

    json json_request;
    if (__rest_server->handle_json(nc, ev, ev_data, json_request) != 0) {
        return;
    }

    connectionInstance conn(nc, ev, ev_data);
    __rest_server->json_callbacks[url](conn, json_request);
}

void restServer::register_json_callback(string endpoint, std::function<void(connectionInstance&, json&) > callback) {
    if (json_callbacks.count(endpoint)) {
        WARN("Call back %s already exists, overriding old call back!!", endpoint.c_str());
    }
    INFO("Adding REST endpoint: %s", endpoint.c_str());
    json_callbacks[endpoint] = callback;
}

int restServer::handle_json(mg_connection* nc, int ev, void* ev_data, json &json_parse) {

    struct http_message *msg = (struct http_message *)ev_data;

    try {
        json_parse = json::parse(string(msg->body.p, msg->body.len));
    } catch (std::exception ex) {
        string message =  string("Error Message: JSON failed to parse, error: ") + string(ex.what());
        mg_send_head(nc, STATUS_BAD_REQUEST, 0, message.c_str());
        return -1;
    }
    return 0;
}

void restServer::mongoose_thread() {

    // init http server
    mg_mgr_init(&mgr, NULL);
    nc = mg_bind(&mgr, port, handle_request);
    if (!nc) {
        INFO("restServer: cannot bind to %s", port);
        return;
    }
    mg_set_protocol_http_websocket(nc);

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

// *** Connection Instance functions ***

connectionInstance::connectionInstance(mg_connection* nc_, int ev_, void* ev_data_) :
    nc(nc_), ev(ev_), ev_data(ev_data_) {
    (void)ev; // No warning
}

connectionInstance::~connectionInstance() {

}

string connectionInstance::get_body() {
    struct http_message *msg = (struct http_message *)ev_data;
    return string(msg->body.p, msg->body.len);
}

string connectionInstance::get_full_message() {
    struct http_message *msg = (struct http_message *)ev_data;
    return string(msg->message.p, msg->message.len);
}

void connectionInstance::send_empty_reply(int status_code) {
    mg_send_head(nc, status_code, 0, NULL);
}

void connectionInstance::send_binary_reply(uint8_t * data, int len) {
    assert(data != nullptr);
    assert(len > 0);

    mg_send_head(nc, STATUS_OK, len, "Content-Type: application/octet-stream");
    mg_send(nc, (void *)data, len);
}

void connectionInstance::send_error(const string& message, int status_code) {
    string error_message = "Error: " + message;
    mg_send_head(nc, status_code, 0, error_message.c_str());
}

void connectionInstance::send_json_reply(json &json_reply) {
    string json_string = json_reply.dump(0);
    mg_send_head(nc, STATUS_OK, json_string.size(), NULL);
    mg_send(nc, (void*) json_string.c_str(), json_string.size());
}
