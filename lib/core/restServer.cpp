#include "restServer.hpp"
#include "errors.h"

#include <pthread.h>
#include <sched.h>
#include <string>
#ifdef MAC_OSX
	#include "osxBindCPU.hpp"
#endif

using json = nlohmann::json;
using std::string;
using std::vector;
using std::map;

restServer &restServer::instance() {
    static restServer server_instance;
    return server_instance;
}

restServer::restServer() : main_thread() {

    stop_thread = false;
    main_thread = std::thread(&restServer::mongoose_thread, this);

#ifndef MAC_OSX
    pthread_setname_np(main_thread.native_handle(), "rest_server");
#endif

    // Framework level tracking of endpoints.
    using namespace std::placeholders;
    register_get_callback("/endpoints",
        std::bind(&restServer::endpoint_list_callback, this, _1));
}

restServer::~restServer() {
    stop_thread = true;
    main_thread.join();
}

void restServer::handle_request(mg_connection* nc, int ev, void* ev_data) {
    if (ev != MG_EV_HTTP_REQUEST)
        return;

    restServer &server = restServer::instance();

    struct http_message *msg = (struct http_message *)ev_data;
    string url = string(msg->uri.p, msg->uri.len);
    string method = string(msg->method.p, msg->method.len);

    map<string, string> &aliases = server.get_aliases();
    if (aliases.find(url) != aliases.end()) {
        url = aliases[url];
    }

    if (method == "GET") {
        if (!server.get_callbacks.count(url)) {
            DEBUG("GET Endpoint %s called, but not found", url.c_str());
            mg_send_head(nc, static_cast<int>(HTTP_RESPONSE::NOT_FOUND), 0, NULL);
            return;
        }
        connectionInstance conn(nc, ev, ev_data);
        server.get_callbacks[url](conn);
        return;
    }

    // TODO should we add `&& Content == Application/JSON` ?
    if (method == "POST") {
        if (!server.json_callbacks.count(url)) {
            DEBUG("Endpoint %s called, but not found", url.c_str());
            mg_send_head(nc, static_cast<int>(HTTP_RESPONSE::NOT_FOUND), 0, NULL);
            return;
        }

        json json_request;
        if (server.handle_json(nc, ev, ev_data, json_request) != 0) {
            return;
        }

        connectionInstance conn(nc, ev, ev_data);
        server.json_callbacks[url](conn, json_request);
        return;
    }

    WARN("Call back with method != POST or GET called, method = %s, endpoint = %s",
         method.c_str(), url.c_str());
    mg_send_head(nc, static_cast<int>(HTTP_RESPONSE::BAD_REQUEST), 0, NULL);
}

void restServer::register_get_callback(string endpoint, std::function<void(connectionInstance&) > callback) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = "/" + endpoint;
    }
    if (get_callbacks.count(endpoint)) {
        WARN("Call back %s already exists, overriding old call back!!", endpoint.c_str());
    }
    INFO("Adding REST endpoint: %s", endpoint.c_str());
    get_callbacks[endpoint] = callback;
}

void restServer::register_post_callback(string endpoint, std::function<void(connectionInstance&, json&) > callback) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = "/" + endpoint;
    }
    if (json_callbacks.count(endpoint)) {
        WARN("Call back %s already exists, overriding old call back!!", endpoint.c_str());
    }
    INFO("Adding REST endpoint: %s", endpoint.c_str());
    json_callbacks[endpoint] = callback;
}

void restServer::remove_get_callback(string endpoint) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = "/" + endpoint;
    }
    auto it = get_callbacks.find(endpoint);
    if (it != get_callbacks.end()) {
        get_callbacks.erase(it);
    }
}

void restServer::remove_json_callback(string endpoint) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = "/" + endpoint;
    }
    auto it = json_callbacks.find(endpoint);
    if (it != json_callbacks.end()) {
        json_callbacks.erase(it);
    }
}

void restServer::add_alias(string alias, string target) {
    if (alias.substr(0, 1) != "/") {
        alias = "/" + alias;
    }
    if (target.substr(0, 1) != "/") {
        target = "/" + target;
    }

    if (json_callbacks.find(alias) != json_callbacks.end() ||
        get_callbacks.find(alias) != get_callbacks.end()) {
        WARN("The endpoint %s already exists, cannot add an alias with that name");
        return;
    }

    aliases[alias] = target;
}

void restServer::remove_alias(string alias) {
    if (alias.substr(0, 1) != "/") {
        alias = "/" + alias;
    }

    auto it = aliases.find(alias);
    if (it != aliases.end()) {
        aliases.erase(it);
    }
}

void restServer::add_aliases_from_config(Config &config) {
    if (!config.exists("/rest_server", "aliases"))
        return;
    json config_aliases = config.get_value("/rest_server", "aliases");
    INFO("%s", config_aliases.dump().c_str());
    for (json::iterator it = config_aliases.begin(); it != config_aliases.end(); ++it) {
        add_alias(it.key(), it.value());
    }
}

void restServer::remove_all_aliases() {
    aliases.clear();
}

map<string, string> &restServer::get_aliases() {
    return aliases;
}

int restServer::handle_json(mg_connection* nc, int ev, void* ev_data, json &json_parse) {

    struct http_message *msg = (struct http_message *)ev_data;

    try {
        json_parse = json::parse(string(msg->body.p, msg->body.len));
    } catch (std::exception ex) {
        string message =  string("Error Message: JSON failed to parse, error: ") + string(ex.what());
        mg_send_head(nc, static_cast<int>(HTTP_RESPONSE::BAD_REQUEST), 0, message.c_str());
        return -1;
    }
    return 0;
}

void restServer::endpoint_list_callback(connectionInstance &conn) {
    json reply;

    vector<string> get_callback_names;
    for (auto &endpoint : get_callbacks) {
        get_callback_names.push_back(endpoint.first);
    }

    vector<string> post_json_callback_names;
    for (auto &endpoint : json_callbacks) {
        post_json_callback_names.push_back(endpoint.first);
    }

    json aliases_names;
    for (auto &item : aliases) {
        aliases_names[item.first] = item.second;
    }

    reply["GET"] = get_callback_names;
    reply["POST"] = post_json_callback_names;
    reply["aliases"] = aliases_names;

    conn.send_json_reply(reply);
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
    while (!stop_thread) mg_mgr_poll(&mgr, 1000);

    mg_mgr_free(&mgr);
}

void restServer::set_server_affinity(Config &config) {
    vector<int32_t> cpu_affinity = config.get_int_array("/rest_server", "cpu_affinity");

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto core_id : cpu_affinity)
        CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(main_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
}

// *** Connection Instance functions ***

connectionInstance::connectionInstance(mg_connection* nc_, int ev_, void* ev_data_) :
    nc(nc_), ev(ev_), ev_data(ev_data_) {
    (void)ev; // No warning
}

connectionInstance::~connectionInstance() {

}

string connectionInstance::get_uri() {
    struct http_message *msg = (struct http_message *)ev_data;
    return string(msg->uri.p, msg->uri.len);
}

string connectionInstance::get_body() {
    struct http_message *msg = (struct http_message *)ev_data;
    return string(msg->body.p, msg->body.len);
}

string connectionInstance::get_full_message() {
    struct http_message *msg = (struct http_message *)ev_data;
    return string(msg->message.p, msg->message.len);
}

void connectionInstance::send_empty_reply(const HTTP_RESPONSE &status) {
    mg_send_head(nc, static_cast<int>(status), 0, NULL);
}

void connectionInstance::send_text_reply(const string &reply, const HTTP_RESPONSE &status) {
    mg_send_head(nc, static_cast<int>(status), reply.length(), "Content-Type: text/plain");
    mg_send(nc, (void *) reply.c_str(), reply.length());
}

void connectionInstance::send_binary_reply(uint8_t * data, int len) {
    assert(data != nullptr);
    assert(len > 0);

    mg_send_head(nc, static_cast<int>(HTTP_RESPONSE::OK),
                 len, "Content-Type: application/octet-stream");
    mg_send(nc, (void *)data, len);
}

void connectionInstance::send_error(const string& message, const HTTP_RESPONSE &status) {
    string error_message = "Error: " + message;
    mg_send_head(nc, static_cast<int>(status), 0, error_message.c_str());
}

void connectionInstance::send_json_reply(json &json_reply) {
    string json_string = json_reply.dump(0);
    mg_send_head(nc, static_cast<int>(HTTP_RESPONSE::OK), json_string.size(), NULL);
    mg_send(nc, (void*) json_string.c_str(), json_string.size());
}
