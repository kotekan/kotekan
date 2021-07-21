#include "restServer.hpp"

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for ERROR_NON_OO, WARN_NON_OO, INFO_NON_OO, DEBUG_NON_OO

#include "fmt.hpp" // for format, fmt

#include <chrono>

#include <algorithm>               // for max
#include <assert.h>                // for assert
#include <cstdint>                 // for int32_t
#include <event2/buffer.h>         // for evbuffer_add, evbuffer_peek, iovec, evbuffer_free
#include <event2/event.h>          // for event_add, event_base_dispatch, event_base_free, even...
#include <event2/http.h>           // for evhttp_send_reply, evhttp_add_header, evhttp_request_...
#include <event2/keyvalq_struct.h> // for evkeyvalq, evkeyval, evkeyval::(anonymous)
#include <event2/thread.h>         // for evthread_use_pthreads
#include <evhttp.h>                // for evhttp_request
#include <exception>               // for exception
#include <mutex>                   // for unique_lock
#include <netinet/in.h>            // for sockaddr_in, ntohs
#include <pthread.h>               // for pthread_setaffinity_np, pthread_setname_np
#include <sched.h>                 // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept>               // for runtime_error
#include <stdlib.h>                // for exit, free, malloc, size_t
#include <string>                  // for string, basic_string, allocator, operator!=, operator+
#include <sys/socket.h>            // for getsockname, socklen_t
#include <sys/time.h>              // for timeval
#include <utility>                 // for pair
#include <vector>                  // for vector
#ifdef MAC_OSX
#include "osxBindCPU.hpp"
#endif

namespace kotekan {

using json = nlohmann::json;
using std::map;
using std::string;
using std::vector;

restServer& restServer::instance() {
    static restServer server_instance;
    return server_instance;
}

restServer::restServer() :
    port(_port),
    main_thread(){
    stop_thread = false;
    timer_metrics = &prometheus::Metrics::instance().add_gauge(
        "kotekan_rest_server_callback_time_miliseconds", "/rest_server", {"endpoint_name", "avg_time", "max_time"});
}

restServer::~restServer() {
    stop_thread = true;
    try {
        main_thread.join();
    } catch (std::exception& e) {
        WARN_NON_OO("restServer: Failure when joining server thread: {:s}", e.what());
        WARN_NON_OO("restServer: Was the server used but never started?");
    }
}

void restServer::start(const std::string& bind_address, u_short port) {

    this->bind_address = bind_address;
    this->_port = port;

    main_thread = std::thread(&restServer::http_server_thread, this);

#ifndef MAC_OSX
    pthread_setname_np(main_thread.native_handle(), "rest_server");
#endif

    // Framework level tracking of endpoints.
    using namespace std::placeholders;
    register_get_callback("/endpoints", std::bind(&restServer::endpoint_list_callback, this, _1));
}

void restServer::stop() {
    timer_metrics = nullptr;
}

void restServer::handle_request(struct evhttp_request* request, void* cb_data) {

    restServer* server = (restServer*)(cb_data);

    string url = string(evhttp_uri_get_path(evhttp_request_get_evhttp_uri(request)));

    DEBUG2_NON_OO("restServer: Got request with url {:s}", url);

    {
        // TODO This function should be locked against changes to the callback
        // maps form other threads.  However there are a number of callbacks (start, stop, etc)
        // which add or remove callbacks from the maps. So a more fine-grained
        // locking system is needed here.
        // std::shared_lock<std::shared_timed_mutex> lock(server->callback_map_lock);
        map<string, string>& aliases = server->get_aliases();
        if (aliases.find(url) != aliases.end()) {
            url = aliases[url];
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        if (request->type == EVHTTP_REQ_GET) {
            connectionInstance conn(request);
            if (!server->get_callbacks.count(url)) {
                DEBUG_NON_OO("restServer: GET Endpoint {:s} called, but not found", url);
                conn.send_error("Not Found", HTTP_RESPONSE::NOT_FOUND);
                return;
            }
            server->get_callbacks[url](conn);

            // Compute callback reply time and save it to stat tracker
            auto t_end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::string endpoint_name = url + "[GET]";
            server->callback_timers[endpoint_name].add_sample(duration);
            double avg = server->callback_timers[endpoint_name].get_avg();
            double max = server->callback_timers[endpoint_name].get_max();
            // server->timer_metrics.labels({endpoint_name, std::to_string(avg), std::to_string(max)}).set(duration);
            return;
        }

        if (request->type == EVHTTP_REQ_POST) {
            connectionInstance conn(request);
            if (!server->json_callbacks.count(url)) {
                DEBUG_NON_OO("restServer: POST Endpoint {:s} called, but not found", url);
                conn.send_error("Not Found", HTTP_RESPONSE::NOT_FOUND);
                return;
            }

            // We currently assume that POST requests come with a JSON message
            json json_request;
            if (server->handle_json(request, json_request) != 0) {
                return;
            }

            server->json_callbacks[url](conn, json_request);

            // Compute callback reply time and save it to stat tracker
            auto t_end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::string endpoint_name = url + "[POST]";
            server->callback_timers[endpoint_name].add_sample(duration);
            double avg = server->callback_timers[endpoint_name].get_avg();
            double max = server->callback_timers[endpoint_name].get_max();
            // server->timer_metrics.labels({endpoint_name, std::to_string(avg), std::to_string(max)}).set(duration);
            return;
        }
    }

    DEBUG_NON_OO("restServer: Call back with method != POST|GET called!");

    connectionInstance conn(request);
    conn.send_error("Bad Request", HTTP_RESPONSE::BAD_REQUEST);
}

void restServer::register_get_callback(string endpoint,
                                       std::function<void(connectionInstance&)> callback) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = fmt::format(fmt("/{:s}"), endpoint);
    }

    {
        std::unique_lock<std::shared_timed_mutex> lock(callback_map_lock);
        if (get_callbacks.count(endpoint)) {
            WARN_NON_OO("restServer: Call back {:s} already exists, overriding old call back!!",
                        endpoint);
        } else {
            std::string endpoint_name = endpoint + "[GET]";
            callback_timers[endpoint_name] = StatTracker();
        }
        get_callbacks[endpoint] = callback;
    }
    INFO_NON_OO("restServer: Adding GET endpoint: {:s}", endpoint);
}

void restServer::register_post_callback(string endpoint,
                                        std::function<void(connectionInstance&, json&)> callback) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = fmt::format(fmt("/{:s}"), endpoint);
    }

    {
        std::unique_lock<std::shared_timed_mutex> lock(callback_map_lock);
        if (json_callbacks.count(endpoint)) {
            WARN_NON_OO("restServer: Callback {:s} already exists, overriding old callback!!",
                        endpoint);
        } else {
            std::string endpoint_name = endpoint + "[POST]";
            callback_timers[endpoint_name] = StatTracker();
        }
        json_callbacks[endpoint] = callback;
    }
    INFO_NON_OO("restServer: Adding POST endpoint: {:s}", endpoint);
}

void restServer::remove_get_callback(string endpoint) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = fmt::format(fmt("/{:s}"), endpoint);
    }

    std::unique_lock<std::shared_timed_mutex> lock(callback_map_lock);
    auto it = get_callbacks.find(endpoint);
    if (it != get_callbacks.end()) {
        get_callbacks.erase(it);
    }
}

void restServer::remove_json_callback(string endpoint) {
    if (endpoint.substr(0, 1) != "/") {
        endpoint = fmt::format(fmt("/{:s}"), endpoint);
    }

    std::unique_lock<std::shared_timed_mutex> lock(callback_map_lock);
    auto it = json_callbacks.find(endpoint);
    if (it != json_callbacks.end()) {
        json_callbacks.erase(it);
    }
}

void restServer::add_alias(string alias, string target) {
    if (alias.substr(0, 1) != "/") {
        alias = fmt::format(fmt("/{:s}"), alias);
    }
    if (target.substr(0, 1) != "/") {
        target = fmt::format(fmt("/{:s}"), target);
    }

    std::unique_lock<std::shared_timed_mutex> lock(callback_map_lock);
    if (json_callbacks.find(alias) != json_callbacks.end()
        || get_callbacks.find(alias) != get_callbacks.end()) {
        WARN_NON_OO("restServer: The endpoint {:s} already exists, cannot add an alias for {:s} "
                    "with that name",
                    alias, target);
        return;
    }
    aliases[alias] = target;
}

void restServer::remove_alias(string alias) {
    if (alias.substr(0, 1) != "/") {
        alias = fmt::format(fmt("/{:s}"), alias);
    }

    std::unique_lock<std::shared_timed_mutex> lock(callback_map_lock);
    auto it = aliases.find(alias);
    if (it != aliases.end()) {
        aliases.erase(it);
    }
}

void restServer::add_aliases_from_config(Config& config) {
    if (!config.exists("/rest_server", "aliases"))
        return;
    json config_aliases = config.get_value("/rest_server", "aliases");
    INFO_NON_OO("restServer: config aliases: {:s}", config_aliases.dump());
    for (json::iterator it = config_aliases.begin(); it != config_aliases.end(); ++it) {
        add_alias(it.key(), it.value());
    }
}

void restServer::remove_all_aliases() {
    std::unique_lock<std::shared_timed_mutex> lock(callback_map_lock);
    aliases.clear();
}

map<string, string>& restServer::get_aliases() {
    return aliases;
}

string restServer::get_http_message(struct evhttp_request* request) {

    string str_data;

    // get input buffer
    evbuffer* input_buffer = evhttp_request_get_input_buffer(request);
    size_t datalen = evbuffer_get_length(input_buffer);
    if (datalen == 0) {
        return "";
    }

    // Reserve space to avoid causing mallocs when appending data.
    str_data.reserve(datalen);

    // peek into the input buffer
    // (treating it as char's and putting it into a string)
    struct evbuffer_iovec* vec_out;
    size_t written = 0;
    // determine how many chunks we need.
    int n_vec = evbuffer_peek(input_buffer, datalen, nullptr, nullptr, 0);
    if (n_vec < 0) {
        WARN_NON_OO("restClient: Failure in evbuffer_peek(), assuming no message and returning an "
                    "empty string");
        return "";
    }

    // Allocate space for the chunks.
    vec_out = (evbuffer_iovec*)malloc(sizeof(struct evbuffer_iovec) * n_vec);
    n_vec = evbuffer_peek(input_buffer, datalen, nullptr, vec_out, n_vec);
    for (int i = 0; i < n_vec; i++) {
        size_t len = vec_out[i].iov_len;
        if (written + len > datalen)
            len = datalen - written;
        str_data.append((char*)vec_out[i].iov_base, len);
        written += len;
    }
    free(vec_out);

    return str_data;
}

int restServer::handle_json(struct evhttp_request* request, json& json_parse) {

    struct evbuffer* ev_buf = evhttp_request_get_input_buffer(request);
    if (ev_buf == nullptr) {
        ERROR_NON_OO("restServer: Cannot get the libevent buffer for the request");
        return -1;
    }

    string message = get_http_message(request);

    if (message.empty()) {
        ERROR_NON_OO("restServer: Request is empty, returning error");
        string error_message = "Error Message: Message was empty, expected JSON string";
        evhttp_send_error(request, static_cast<int>(HTTP_RESPONSE::BAD_REQUEST), message.c_str());
        return -1;
    }

    try {
        json_parse = json::parse(message);
    } catch (const std::exception& ex) {
        string error_message =
            string("Error Message: JSON failed to parse, error: ") + string(ex.what());
        ERROR_NON_OO(
            "restServer: Failed to pase JSON from request, the error is '{:s}', and the HTTP "
            "message was: {:s}",
            ex.what(), message);
        evhttp_send_error(request, static_cast<int>(HTTP_RESPONSE::BAD_REQUEST),
                          error_message.c_str());
        return -1;
    }
    return 0;
}

void restServer::endpoint_list_callback(connectionInstance& conn) {
    json reply;

    vector<string> get_callback_names;
    for (auto& endpoint : get_callbacks) {
        get_callback_names.push_back(endpoint.first);
    }

    vector<string> post_json_callback_names;
    for (auto& endpoint : json_callbacks) {
        post_json_callback_names.push_back(endpoint.first);
    }

    json aliases_names;
    for (auto& item : aliases) {
        aliases_names[item.first] = item.second;
    }

    reply["GET"] = get_callback_names;
    reply["POST"] = post_json_callback_names;
    reply["aliases"] = aliases_names;

    conn.send_json_reply(reply);
}

void restServer::timer(evutil_socket_t fd, short event, void* arg) {

    // Unused parameters, required by libevent. Suppress warning.
    (void)fd;
    (void)event;

    restServer* rest_server = (restServer*)arg;
    if (rest_server->stop_thread) {
        event_base_loopbreak(rest_server->event_base);
    }
}

void restServer::http_server_thread() {

    // Allow for using extra threads (not currently needed)
    if (evthread_use_pthreads()) {
        ERROR_NON_OO("restServer: Cannot use pthreads with libevent!");
        return;
    }

    // Create the base event for handling requests
    event_base = event_base_new();
    if (event_base == nullptr) {
        ERROR_NON_OO("restServer: Failed to create libevent base");
        // Use exit() not raise() since this happens early in startup before
        // the signal handlers are all in place.
        exit(1);
    }

    // Create the server
    ev_server = evhttp_new(event_base);
    if (ev_server == nullptr) {
        ERROR_NON_OO("restServer: Failed to create libevent base");
        exit(1);
    }

    // Currently allow only GET and POST requests
    evhttp_set_allowed_methods(ev_server, EVHTTP_REQ_GET | EVHTTP_REQ_POST);

    // Just setup one handler and implement the URL parsing internally
    evhttp_set_gencb(ev_server, handle_request, (void*)this);

    // Bind to the IP and port
    struct evhttp_bound_socket* ev_sock =
        evhttp_bind_socket_with_handle(ev_server, bind_address.c_str(), _port);
    if (ev_sock == nullptr) {
        ERROR_NON_OO("restServer: Failed to bind to {:s}:{:d}", bind_address, _port);
        exit(1);
    }

    // if port was set to random, find port socket is listening on
    if (_port == 0) {
        evutil_socket_t sock = evhttp_bound_socket_get_fd(ev_sock);
        struct sockaddr_in sin;
        socklen_t len = sizeof(sin);
        if (getsockname(sock, (struct sockaddr*)&sin, &len) == -1) {
            ERROR_NON_OO("restServer: Failed getting socket name ({:s}:{:d})", bind_address, _port);
            exit(1);
        }
        _port = ntohs(sin.sin_port);
    }
    // This INFO line is parsed by the python runner to get the RESTserver port. Don't edit.
    INFO_NON_OO("restServer: started server on address:port {:s}:{:d}", bind_address, _port);

    // Create a timer to check for the exit condition
    struct event* timer_event;
    timer_event = event_new(event_base, -1, EV_PERSIST, &restServer::timer, this);
    struct timeval interval;
    interval.tv_sec = 0;
    interval.tv_usec = 100000;
    event_add(timer_event, &interval);

    // run event loop
    event_base_dispatch(event_base);

    evhttp_free(ev_server);
    event_base_free(event_base);
}

void restServer::set_server_affinity(Config& config) {
    vector<int32_t> cpu_affinity = config.get<std::vector<int32_t>>("/rest_server", "cpu_affinity");

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto core_id : cpu_affinity)
        CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(main_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
}

string restServer::get_http_responce_code_text(const HTTP_RESPONSE& status) {
    switch (status) {
        case HTTP_RESPONSE::OK:
            return "OK";
        case HTTP_RESPONSE::NOT_FOUND:
            return "NOT_FOUND";
        case HTTP_RESPONSE::INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case HTTP_RESPONSE::BAD_REQUEST:
            return "BAD_REQUEST";
        case HTTP_RESPONSE::REQUEST_FAILED:
            return "REQUEST_FAILED";
        default:
            return "";
    }
}

// *** Connection Instance functions ***

connectionInstance::connectionInstance(struct evhttp_request* request) : request(request) {
    event_buffer = evbuffer_new();
    if (event_buffer == nullptr) {
        throw std::runtime_error("Failed to create evbuffer");
    }
}

connectionInstance::~connectionInstance() {
    evbuffer_free(event_buffer);
}

string connectionInstance::get_uri() {
    return string(evhttp_request_get_uri(request));
}

string connectionInstance::get_body() {
    return restServer::get_http_message(request);
}

void connectionInstance::send_empty_reply(const HTTP_RESPONSE& status) {
    evhttp_send_reply(request, static_cast<int>(status),
                      restServer::get_http_responce_code_text(status).c_str(), event_buffer);
}

void connectionInstance::send_text_reply(const string& reply_message) {

    if (evhttp_add_header(evhttp_request_get_output_headers(request), "Content-Type", "text/plain")
        != 0) {
        throw std::runtime_error("Failed to add header to reply");
    }

    if (evbuffer_add(event_buffer, (void*)reply_message.c_str(), reply_message.size()) != 0) {
        throw std::runtime_error("Failed to add reply message");
    }

    evhttp_send_reply(request, static_cast<int>(HTTP_RESPONSE::OK), "OK", event_buffer);
}

void connectionInstance::send_binary_reply(uint8_t* data, int len) {
    assert(data != nullptr);
    assert(len > 0);

    if (evhttp_add_header(evhttp_request_get_output_headers(request), "Content-Type",
                          "Application/octet-stream")
        != 0) {
        throw std::runtime_error("Failed to add header to reply");
    }

    if (evbuffer_add(event_buffer, (void*)data, len) != 0) {
        throw std::runtime_error("Failed to add data to reply message");
    }

    evhttp_send_reply(request, static_cast<int>(HTTP_RESPONSE::OK), "OK", event_buffer);
}

void connectionInstance::send_error(const string& message, const HTTP_RESPONSE& status) {
    if (evhttp_add_header(evhttp_request_get_output_headers(request), "Content-Type",
                          "Application/JSON")
        != 0) {
        throw std::runtime_error("Failed to add header to reply");
    }

    string reply = json{{"message", message}, {"code", status}}.dump();
    if (evbuffer_add(event_buffer, (void*)reply.c_str(), reply.size()) != 0) {
        throw std::runtime_error("Failed to add reply message");
    }

    evhttp_send_reply(request, static_cast<int>(status),
                      restServer::get_http_responce_code_text(status).c_str(), event_buffer);
}

void connectionInstance::send_json_reply(const json& json_reply) {
    string json_string = json_reply.dump(0);

    if (evhttp_add_header(evhttp_request_get_output_headers(request), "Content-Type",
                          "Application/JSON")
        != 0) {
        throw std::runtime_error("Failed to add header to reply");
    }

    if (evbuffer_add(event_buffer, (void*)json_string.c_str(), json_string.size()) != 0) {
        throw std::runtime_error("Failed to add JSON string to reply message");
    }

    evhttp_send_reply(request, static_cast<int>(HTTP_RESPONSE::OK), "OK", event_buffer);
}

std::map<std::string, std::string> connectionInstance::get_query() {
    std::map<std::string, std::string> query_map;
    struct evkeyvalq queries;
    queries.tqh_first = nullptr;
    queries.tqh_last = nullptr;
    const char* query_string = evhttp_uri_get_query(evhttp_request_get_evhttp_uri(request));
    if (query_string && evhttp_parse_query_str(query_string, &queries) == 0) {
        struct evkeyval* cur_query = queries.tqh_first;
        while (cur_query) {
            query_map[string(cur_query->key)] = string(cur_query->value);
            cur_query = cur_query->next.tqe_next;
        }
    }
    evhttp_clear_headers(&queries);
    return query_map;
}

} // namespace kotekan
