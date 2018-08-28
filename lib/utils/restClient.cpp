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

restReply restClient::_reply = { false, nullptr, 0 };

void restClient::http_request_done(struct evhttp_request *req, void *arg){
    _reply.datalen = 0;
    _reply.data = nullptr;

    if (req == NULL) {
        int errcode = EVUTIL_SOCKET_ERROR();
        WARN("restClient: request failed.");
        // Print socket error
        WARN("socket error = %s (%d)\n",
             evutil_socket_error_to_string(errcode), errcode);
        return;
    }

    int response_code = evhttp_request_get_response_code(req);

    if (response_code == 200)
        _reply.success = true;
    else {
        INFO("restClient: Received respose code %d", response_code);
        _reply.success = false;
    }

    // get input buffer
    evbuffer* input_buffer = evhttp_request_get_input_buffer(req);
    _reply.datalen = evbuffer_get_length(input_buffer);

    // reserve memory
    _reply.data = new char[_reply.datalen];
    if (_reply.data == nullptr) {
        WARN("restClient: Failure reserving memory for received data.");
        _reply.success = false;
        return;
    }

    // copy data out of the input buffer
    if (evbuffer_remove(input_buffer, _reply.data, _reply.datalen) == -1) {
        WARN("restClient: Unable to drain input buffer.");
        _reply.success = false;
        _reply.datalen = 0;
    }
}

std::unique_ptr<restReply> restClient::send(std::string path,
                                            const nlohmann::json& data,
                                            const std::string& host,
                                            const unsigned short port,
                                            const int retries,
                                            const int timeout) {
    struct event_base *base;
    struct evhttp_connection *evcon = NULL;
    struct evhttp_request *req;
    struct evkeyvalq *output_headers;
    struct evbuffer *output_buffer;

    size_t datalen = data.dump().length() + 1;
    char json_string[datalen];

    int ret;

    // Fix path in case it is nothing or missing '/' in the beginning
    if (path.length() == 0) {
        path = string("/");
    } else if (path.at(0) != '/')
        path = "/" + path;

    base = event_base_new();
    if (!base) {
        WARN("restClient: Failure creating new event_base.");
        _reply.success = false;
        return std::make_unique<restReply>(_reply);
    }

    // DNS resolution is blocking (if not numeric host is passed)
    evcon = evhttp_connection_base_new(base, NULL, host.c_str(), port);
    if (evcon == NULL) {
        WARN("restClient: evhttp_connection_base_new() failed.");
        _reply.success = false;
        return std::make_unique<restReply>(_reply);
    }

    if (retries > 0) {
        evhttp_connection_set_retries(evcon, retries);
    }
    if (timeout >= 0) {
        evhttp_connection_set_timeout(evcon, timeout);
    }

    // Fire off the request
    req = evhttp_request_new(http_request_done, NULL);
    if (req == NULL) {
        WARN("restClient: evhttp_request_new() failed.");
        _reply.success = false;
        return std::make_unique<restReply>(_reply);
    }

    //conn = evhttp_connection_base_new(base, NULL, host.c_str(), port);
    //req = evhttp_request_new(http_request_done, base);

    output_headers = evhttp_request_get_output_headers(req);
    ret = evhttp_add_header(output_headers, "Host", host.c_str());
    if (ret) {
        WARN("restClient: Failure adding \"Host\" header.");
        _reply.success = false;
        return std::make_unique<restReply>(_reply);
    }
    ret = evhttp_add_header(output_headers, "Connection", "close");
    if (ret) {
        WARN("restClient: Failure adding \"Connection\" header.");
        _reply.success = false;
        return std::make_unique<restReply>(_reply);
    }
    ret = evhttp_add_header(output_headers, "Content-Type", "application/json");
    if (ret) {
        WARN("restClient: Failure adding \"Content-Type\" header.");
        _reply.success = false;
        return std::make_unique<restReply>(_reply);
    }

    if (!data.empty()) {
        char buf[256];
        strcpy(json_string, data.dump().c_str());

        output_buffer = evhttp_request_get_output_buffer(req);
        ret = evbuffer_add_reference(output_buffer, json_string, datalen, NULL,
                                     NULL);
        if (ret) {
            WARN("restClient: Failure adding JSON data to event buffer.");
            _reply.success = false;
            return std::make_unique<restReply>(_reply);
        }

        evutil_snprintf(buf, sizeof(buf)-1, "%lu", (unsigned long)datalen);
        evhttp_add_header(output_headers, "Content-Length", buf);
        if (ret) {
            WARN("restClient: Failure adding \"Content-Length\" header.");
            _reply.success = false;
            return std::make_unique<restReply>(_reply);
        }
        DEBUG("restClient: Sending %s bytes: %s", buf, data.dump().c_str());

        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_POST, path.c_str());
    } else {
        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_GET, path.c_str());
    }
    if (ret) {
        WARN("restClient: evhttp_make_request() failed.");
        _reply.success = false;
        return std::make_unique<restReply>(_reply);
    }

    ret = event_base_dispatch(base);
    if (ret < 0) {
        _reply.success = false;
        WARN("restClient::send: Failure sending message %s to %s:%d%s.",
             json_string, host.c_str(), port, path.c_str());
    }

    // Cleanup
    if (evcon)
        evhttp_connection_free(evcon);
    event_base_free(base);

    return std::make_unique<restReply>(_reply);
}
