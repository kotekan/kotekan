#include "restClient.hpp"
#include "errors.h"

#include <evhttp.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/dns.h>
#include <event2/thread.h>
#include <cxxabi.h>


restClient &restClient::instance() {
    static restClient client_instance;
    return client_instance;
}

restClient::restClient() : _main_thread() {

    _stop_thread = false;
    _main_thread = std::thread(&restClient::event_thread, this);

#ifndef MAC_OSX
    pthread_setname_np(_main_thread.native_handle(), "rest_client");
#endif

    // restClient::instance() was just called the first time.
    // wait a bit to make sure the event_base is initialized in the
    // event_thread before someone makes a request.
    usleep(5000);
}

restClient::~restClient() {
    _stop_thread = true;
    if (event_base_loopbreak(_base))
        ERROR("restClient: event_base_loopbreak() failed.");
    _main_thread.join();
    DEBUG("restClient: event thread stopped.");
}

void restClient::timer(evutil_socket_t fd, short event, void *arg) {
    restClient * client = (restClient *)arg;
    if (client->_stop_thread) {
        event_base_loopbreak(client->_base);
    }
}

void restClient::event_thread() {
    INFO("restClient: libevent version: %s", event_get_version());

    if (evthread_use_pthreads()) {
        ERROR("restClient: Cannot use pthreads with libevent!");
        raise(SIGINT);
    }

    // event base and dns base
    _base = event_base_new();
    if (!_base) {
        ERROR("restClient: Failure creating new event_base.");
        raise(SIGINT);
    }

    // DNS resolution is blocking (if not numeric host is passed)
    _dns = evdns_base_new(_base, 1);
    if (_dns == nullptr) {
        ERROR("restClient: evdns_base_new() failed.");
        raise(SIGINT);
    }

    // Create a timer to check for the exit condition
    struct event *timer_event;
    timer_event = event_new(_base, -1, EV_PERSIST, timer, this);
    struct timeval interval;
    interval.tv_sec = 0;
    interval.tv_usec = 500000;
    event_add(timer_event, &interval);

    // run event loop
    DEBUG("restClient: starting event loop");
    while (!_stop_thread) {
        if(event_base_dispatch(_base) < 0) {
            ERROR("restClient::event_thread(): Failure in the event loop.");
            raise(SIGINT);
        }
    }
    DEBUG("restClient: exiting event loop");

    // Cleanup
    event_free(timer_event);
    evdns_base_free(_dns, 1);
    event_base_free(_base);
}

void restClient::http_request_done(struct evhttp_request *req, void *arg){
    // FIXME: evcon is passed here, because evhttp_request_get_connection(req)
    // doesn't work (libevent 2.0.* problem?)
    // TODO: maybe keep the evhttp_connections in a pool and reuse them
    // (set Connection:keep-alive header)
    auto pair = (std::pair<std::function<void(restReply)>,
                 struct evhttp_connection*>*) arg;
    std::function<void(restReply)> ext_cb = pair->first;

    // this is where we store the reply
    std::string str_data("");

    if (req == nullptr) {
        int errcode = EVUTIL_SOCKET_ERROR();
        WARN("restClient: request failed.");
        // Print socket error
        std::string str = evutil_socket_error_to_string(errcode);
        WARN("restClient: socket error = %s (%d)",
             str.c_str(), errcode);
        ext_cb(restReply(false, str_data));
        cleanup(pair);
        return;
    }

    int response_code = evhttp_request_get_response_code(req);

    if (response_code != 200) {
        // TODO: change to use
        // evhttp_request_get_response_code_line(req) (libevent 2.1)
        INFO("restClient: Received response code %d (%s)", response_code,
             req->response_code_line);
        if (response_code == 0)
            WARN("restClient: connection error.");
        ext_cb(restReply(false, str_data));
        cleanup(pair);
        return;
    }

    // get input buffer
    evbuffer* input_buffer = evhttp_request_get_input_buffer(req);
    size_t datalen = evbuffer_get_length(input_buffer);
    if (datalen == 0) {
        ext_cb(restReply(true, str_data));
        cleanup(pair);
        return;
    }

    // peek into the input buffer
    // (treating it as char's and putting it into a string)
    struct evbuffer_iovec* vec_out;
    size_t written = 0;
    // determine how many chunks we need.
    int n_vec = evbuffer_peek(input_buffer, datalen, NULL, NULL, 0);
    if (n_vec < 0) {
        WARN("restClient: Failure in evbuffer_peek()");
        ext_cb(restReply(false, str_data));
        cleanup(pair);
        return;
    }

    // Allocate space for the chunks.
    vec_out = (iovec*)malloc(sizeof(struct evbuffer_iovec) * n_vec);

    n_vec = evbuffer_peek(input_buffer, datalen, NULL, vec_out, n_vec);
    for (int i = 0; i < n_vec; i++) {
        size_t len = vec_out[i].iov_len;
        if (written + len > datalen)
            len = datalen - written;
        str_data.append((char*)vec_out[i].iov_base, len);
        written += len;
    }
    free(vec_out);

    // call the external callback
    ext_cb(restReply(true, str_data));
    cleanup(pair);
}

void restClient::cleanup(std::pair<std::function<void(restReply)>,
                              struct evhttp_connection*>* pair) {
    if (pair->second)
        evhttp_connection_free(pair->second);
    delete pair;
}

bool restClient::make_request(std::string path,
                              std::function<void(restReply)>
                                request_done_cb,
                              const nlohmann::json& data,
                              const std::string& host,
                              const unsigned short port,
                              const int retries,
                              const int timeout) {
    struct evhttp_connection* evcon = nullptr;
    struct evhttp_request* req;
    struct evkeyvalq *output_headers;
    struct evbuffer *output_buffer;

    int ret;

    // Fix path in case it is nothing or missing '/' in the beginning
    if (path.length() == 0) {
        path = string("/");
    } else if (path.at(0) != '/')
        path = "/" + path;

    evcon = evhttp_connection_base_new(_base, _dns, host.c_str(), port);
    if (evcon == nullptr) {
        WARN("restClient: evhttp_connection_base_new() failed.");
        return false;
    }

    if (retries > 0) {
        evhttp_connection_set_retries(evcon, retries);
    }
    if (timeout >= 0) {
        evhttp_connection_set_timeout(evcon, timeout);
    }

    // Fire off the request and pass the external callback to the internal one.
    // check if external callback function is callable
    if(!request_done_cb) {
        ERROR("restClient: external callback function is not callable.");
        evhttp_connection_free(evcon);
        return false;
    }

    //Test if callback function is of correct type (doesn't compile on MacOS)
#ifndef MAC_OSX
    if (request_done_cb.target_type() != typeid(void(*)(restReply)) &&
        request_done_cb.target_type() != typeid(
            std::_Bind<void (*(std::_Placeholder<1>))
            (std::pair<bool, std::string&>)>)) {
        std::string type(request_done_cb.target_type().name());
        int status;
        ERROR("restClient: external callback function is not of type " \
              "void(*)(restReply), but %s.",
              abi::__cxa_demangle(type.c_str(), 0, 0, &status));
        evhttp_connection_free(evcon);
        return false;
    }
#endif

    // keep the external callback function object on the heap, so it is allowed
    // to run out of scope on the calling side
    // also pass the connection to the callback, so it can be freed there
    std::pair<std::function<void(restReply)>, struct evhttp_connection*>* pair =
            new std::pair<std::function<void(restReply)>,
                          struct evhttp_connection*>
            (request_done_cb, evcon);

    req = evhttp_request_new(http_request_done, pair);
    if (req == nullptr) {
        WARN("restClient: evhttp_request_new() failed.");
        evhttp_connection_free(evcon);
        return false;
    }

    output_headers = evhttp_request_get_output_headers(req);
    ret = evhttp_add_header(output_headers, "Host", host.c_str());
    if (ret) {
        WARN("restClient: Failure adding \"Host\" header.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        return false;
    }
    ret = evhttp_add_header(output_headers, "Connection", "close");
    if (ret) {
        WARN("restClient: Failure adding \"Connection\" header.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        return false;
    }
    ret = evhttp_add_header(output_headers, "Content-Type", "application/json");
    if (ret) {
        WARN("restClient: Failure adding \"Content-Type\" header.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        return false;
    }

    if (!data.empty()) {
        size_t datalen = data.dump().length();
        char buf[256];

        output_buffer = evhttp_request_get_output_buffer(req);

        // copy data into the buffer
        ret = evbuffer_add(output_buffer, data.dump().c_str(), datalen);
        if (ret) {
            WARN("restClient: Failure adding JSON data to event buffer.");
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            return false;
        }

        evutil_snprintf(buf, sizeof(buf)-1, "%lu", (unsigned long)datalen);
        evhttp_add_header(output_headers, "Content-Length", buf);
        if (ret) {
            WARN("restClient: Failure adding \"Content-Length\" header.");
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            return false;
        }
        DEBUG2("restClient: Sending %s bytes: %s", buf, data.dump().c_str());

        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_POST, path.c_str());
    } else {
        DEBUG2("restClient: sending GET request.");
        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_GET, path.c_str());
    }
    if (ret) {
        WARN("restClient: evhttp_make_request() failed.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        return false;
    }
    return true;
}