#include "restClient.hpp"
#include "errors.h"
#include "json.hpp"
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <evhttp.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/bufferevent.h>
#include <event2/dns.h>

bool restClient::_success = false;
size_t restClient::_datalen = 0;
char* restClient::_data = nullptr;

void restClient::http_request_done(struct evhttp_request *req, void *arg){

    _success = false;
    _datalen = 0;
    _data = nullptr;

    if (req == nullptr) {
        int errcode = EVUTIL_SOCKET_ERROR();
        WARN("restClient: request failed.");
        // Print socket error
        std::string str = evutil_socket_error_to_string(errcode);
        WARN("restClient: socket error = %s (%d)",
             str.c_str(), errcode);
        return;
    }

    int response_code = evhttp_request_get_response_code(req);

    if (response_code == 200)
        _success = true;
    else {
        INFO("restClient: Received response code %d", response_code);
    }

    // get input buffer
    evbuffer* input_buffer = evhttp_request_get_input_buffer(req);
    _datalen = evbuffer_get_length(input_buffer);

    // reserve memory
    _data = new char[_datalen];
    if (_data == nullptr) {
        WARN("restClient: Failure reserving memory for received data.");
        _success = false;
        _datalen = 0;
        return;
    }

    // copy data out of the input buffer
    if (evbuffer_remove(input_buffer, _data, _datalen) == -1) {
        WARN("restClient: Unable to drain input buffer.");
        _success = false;
        _datalen = 0;
        _data = nullptr;
    }
}

bool restClient::send(std::string path,
                      const nlohmann::json& data,
                      const std::string& host,
                      const unsigned short port,
                      const int retries,
                      const int timeout) {
    struct event_base* base;
    struct evhttp_connection* evcon = nullptr;
    struct evhttp_request* req;
    struct evdns_base* dns;
    struct evkeyvalq *output_headers;
    struct evbuffer *output_buffer;

    size_t datalen = data.dump().length() + 1;
    char json_string[datalen];

    int ret;

    _data = nullptr;
    _datalen = 0;

    // Fix path in case it is nothing or missing '/' in the beginning
    if (path.length() == 0) {
        path = string("/");
    } else if (path.at(0) != '/')
        path = "/" + path;

    base = event_base_new();
    if (!base) {
        WARN("restClient: Failure creating new event_base.");
        return false;
    }

    // DNS resolution is blocking (if not numeric host is passed)
    dns = evdns_base_new(base, 1);
    if (dns == nullptr) {
        WARN("restClient: evdns_base_new() failed.");
        return false;
    }
    evcon = evhttp_connection_base_new(base, dns, host.c_str(), port);
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

    // Fire off the request
    req = evhttp_request_new(http_request_done, base);
    if (req == nullptr) {
        WARN("restClient: evhttp_request_new() failed.");
        return false;
    }

    output_headers = evhttp_request_get_output_headers(req);
    ret = evhttp_add_header(output_headers, "Host", host.c_str());
    if (ret) {
        WARN("restClient: Failure adding \"Host\" header.");
        return false;
    }
    ret = evhttp_add_header(output_headers, "Connection", "close");
    if (ret) {
        WARN("restClient: Failure adding \"Connection\" header.");
        return false;
    }
    ret = evhttp_add_header(output_headers, "Content-Type", "application/json");
    if (ret) {
        WARN("restClient: Failure adding \"Content-Type\" header.");
        return false;
    }

    if (!data.empty()) {
        char buf[256];
        strcpy(json_string, data.dump().c_str());

        output_buffer = evhttp_request_get_output_buffer(req);
        ret = evbuffer_add_reference(output_buffer, json_string, datalen,
                                     nullptr, nullptr);
        if (ret) {
            WARN("restClient: Failure adding JSON data to event buffer.");
            return false;
        }

        evutil_snprintf(buf, sizeof(buf)-1, "%lu", (unsigned long)datalen);
        evhttp_add_header(output_headers, "Content-Length", buf);
        if (ret) {
            WARN("restClient: Failure adding \"Content-Length\" header.");
            return false;
        }
        DEBUG("restClient: Sending %s bytes: %s", buf, data.dump().c_str());

        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_POST, path.c_str());
    } else {
        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_GET, path.c_str());
    }
    if (ret) {
        WARN("restClient: evhttp_make_request() failed.");
        return false;
    }

    // Cleanup
    evdns_base_free(dns, 1);
    ret = event_base_dispatch(base);
    if (ret < 0) {
        WARN("restClient::send: Failure sending message %s to %s:%d%s.",
             json_string, host.c_str(), port, path.c_str());
        return false;
    }

    evhttp_connection_free(evcon);
    event_base_free(base);

    return _success;
}


std::string restClient::get_reply() {
    if (_success == false) {
        std::string reply(nullptr, 0);

        return reply;
    }

    std::string reply(_data, _datalen);

    return reply;
}