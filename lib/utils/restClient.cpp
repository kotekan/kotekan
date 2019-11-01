#include "restClient.hpp"

#include "errors.h"
#include "signal.h"

#include <chrono>
#include <condition_variable>
#include <event2/buffer.h>
#include <event2/dns.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/thread.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/uio.h>


restClient& restClient::instance() {
    static restClient client_instance;
    return client_instance;
}

restClient::restClient() : _main_thread() {

    _stop_thread = false;
    _event_thread_started = false;
    _main_thread = std::thread(&restClient::event_thread, this);

#ifndef MAC_OSX
    pthread_setname_np(_main_thread.native_handle(), "rest_client");
#endif

    // restClient::instance() was just called the first time.
    // wait until the event_base is initialized in the
    // event_thread before someone makes a request.
    std::unique_lock<std::mutex> lck_start(_mtx_start);
    _cv_start.wait(lck_start, [this]() { return _event_thread_started; });
}

restClient::~restClient() {
    _stop_thread = true;
    if (event_base_loopbreak(_base))
        ERROR_NON_OO("restClient: event_base_loopbreak() failed.");
    _main_thread.join();
    DEBUG_NON_OO("restClient: event thread stopped.");
}

void restClient::timer(evutil_socket_t fd, short event, void* arg) {

    // Unused parameters, required by libevent. Suppress warning.
    (void)fd;
    (void)event;

    restClient* client = (restClient*)arg;
    if (client->_stop_thread) {
        event_base_loopbreak(client->_base);
    }
}

void restClient::event_thread() {
    INFO_NON_OO("restClient: libevent version: {:s}", event_get_version());

    if (evthread_use_pthreads()) {
        FATAL_ERROR_NON_OO("restClient: Cannot use pthreads with libevent!");
    }

    // event base and dns base
    _base = event_base_new();
    if (!_base) {
        FATAL_ERROR_NON_OO("restClient: Failure creating new event_base.");
    }

    // The event loop will run in this seperate thread. We have to schedule requests from
    // this same thread. Therefor we create a bufferevent pair here to pass request pointers in.
    bufferevent* pair[2];
    if (bufferevent_pair_new(_base, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE, pair))
        FATAL_ERROR_NON_OO("restClient: Failure creating bufferevent pair.");
    bev_req_write = pair[0];
    bev_req_read = pair[1];
    if (bufferevent_disable(bev_req_write, EV_READ))
        FATAL_ERROR_NON_OO("restClient: Failure in bufferevent_disable().");
    if (bufferevent_enable(bev_req_read, EV_READ))
        FATAL_ERROR_NON_OO("restClient: Failure in bufferevent_enable().");
    bufferevent_setcb(bev_req_read, _bev_req_readcb, NULL, _bev_req_errcb, this);
    bufferevent_setcb(bev_req_write, NULL, NULL, _bev_req_errcb, this);

    // DNS resolution is blocking (if not numeric host is passed)
    _dns = evdns_base_new(_base, 1);
    if (_dns == nullptr) {
        FATAL_ERROR_NON_OO("restClient: evdns_base_new() failed.");
    }

    // Create a timer to check for the exit condition
    event* timer_event;
    timer_event = event_new(_base, -1, EV_PERSIST, timer, this);
    timeval interval;
    interval.tv_sec = 0;
    interval.tv_usec = 500000;
    event_add(timer_event, &interval);

    {
        std::unique_lock<std::mutex> lck_start(_mtx_start);
        _event_thread_started = true;
    }
    _cv_start.notify_one();

    // run event loop
    DEBUG_NON_OO("restClient: starting event loop");
    while (!_stop_thread) {
        if (event_base_dispatch(_base) < 0) {
            FATAL_ERROR_NON_OO("restClient::event_thread(): Failure in the event loop.");
        }
    }
    DEBUG_NON_OO("restClient: exiting event loop");

    // Cleanup
    bufferevent_free(bev_req_write);
    bufferevent_free(bev_req_read);
    event_free(timer_event);
    evdns_base_free(_dns, 1);
    event_base_free(_base);
}

void restClient::_bev_req_readcb(struct bufferevent* bev, void* arg) {
    (void)bev;

    // Get the restClient object from the arg void pointer
    restClient* client = (restClient*)arg;
    if (!client)
        FATAL_ERROR_NON_OO("restClient: _bev_req_readcb got nullptr client");

    // Sometime libevent calls this cb just once when there are multiple requests in the
    // bufferevent pair. Read until we have all of them.
    int read_size;
    while (true) {
        restRequest* request = nullptr;
        read_size = bufferevent_read(client->bev_req_read, &request, sizeof(restRequest*));
        if (read_size <= 0)
            return;
        if (!request)
            FATAL_ERROR_NON_OO("restClient: _bev_req_readcb got nullptr request");

        DEBUG2_NON_OO("restClient: _bev_req_readcb read {} bytes", read_size);

        if (!client->_make_request(request))
            FATAL_ERROR_NON_OO("restClient: Failure making request.");
    }
}

void restClient::http_request_done(struct evhttp_request* req, void* arg) {
    // FIXME: evcon is passed here, because evhttp_request_get_connection(req)
    // always returns NULL and there is no way to free the connection on
    // completion in libevent < 2.1
    // libevent 2.1 has a evhttp_connection_free_on_completion, but using it,
    // the bufferevent never gets deleted...
    // TODO: maybe keep the evhttp_connections in a pool and reuse them
    // (set Connection:keep-alive header)
    auto pair = (std::pair<std::function<void(restReply)>, struct evhttp_connection*>*)arg;
    std::function<void(restReply)> ext_cb = pair->first;

    // this is where we store the reply
    std::string str_data("");

    if (req == nullptr) {
        int errcode = EVUTIL_SOCKET_ERROR();
        WARN_NON_OO("restClient: request failed.");
        // Print socket error
        std::string str = evutil_socket_error_to_string(errcode);
        WARN_NON_OO("restClient: socket error = {:s} ({:d})", str, errcode);
        ext_cb(restReply(false, str_data));
        cleanup(pair);
        return;
    }

    int response_code = evhttp_request_get_response_code(req);

    if (response_code != 200) {
        std::string status_text = "";
        if (req->response_code_line)
            status_text = req->response_code_line;
        if (response_code == 0)
            status_text = "Connection error";
        INFO_NON_OO("restClient: Received response code {:d} ({:s})", response_code, status_text);
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

    // Reserve space to avoid causing mallocs when appending data.
    str_data.reserve(datalen);

    // peek into the input buffer
    // (treating it as char's and putting it into a string)
    struct evbuffer_iovec* vec_out;
    size_t written = 0;
    // determine how many chunks we need.
    int n_vec = evbuffer_peek(input_buffer, datalen, NULL, NULL, 0);
    if (n_vec < 0) {
        WARN_NON_OO("restClient: Failure in evbuffer_peek()");
        (*ext_cb)(restReply(false, str_data));
        cleanup(pair);
        return;
    }

    // Allocate space for the chunks.
    vec_out = (iovec*)malloc(sizeof(evbuffer_iovec) * n_vec);

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

void restClient::cleanup(
    std::pair<std::function<void(restReply)>, struct evhttp_connection*>* pair) {
    if (pair->second)
        evhttp_connection_free(pair->second);
    delete pair;
}

void restClient::make_request(const std::string& path,
                              std::function<void(restReply)> request_done_cb,
                              const nlohmann::json& data, const std::string& host,
                              const unsigned short port, const int retries, const int timeout) {
    if (!bev_req_write)
        FATAL_ERROR_NON_OO("restClient: make_request called, but bev_req_write is NULL.");

    // create a request object on the heap to let it live until we have a reply...
    restRequest* request =
        new restRequest(path, request_done_cb, data, host, port, retries, timeout);
    if (!request)
        FATAL_ERROR_NON_OO("restClient: memory for restRequest failed");

    // and drop it in the socket bufferevent for the event thread
    if (bufferevent_write(bev_req_write, &request, sizeof(restRequest*)))
        FATAL_ERROR_NON_OO("restClient: Failure writing to socket bufferevent.");
}

bool restClient::_make_request(const struct restRequest* request) {
    struct evhttp_connection* evcon = nullptr;
    struct evhttp_request* req;
    struct evkeyvalq* output_headers;

    int ret;

    evcon = evhttp_connection_base_new(_base, _dns, request->host->c_str(), request->port);
    if (evcon == nullptr) {
        WARN_NON_OO("restClient: evhttp_connection_base_new() failed.");
        delete request;
        return false;
    }

    if (request->retries > 0)
        evhttp_connection_set_retries(evcon, request->retries);
    if (request->timeout >= 0)
        evhttp_connection_set_timeout(evcon, request->timeout);

    // Fire off the request and pass the external callback to the internal one.
    // check if external callback function is callable
    if (!request->request_done_cb) {
        ERROR_NON_OO("restClient: external callback function is not callable.");
        evhttp_connection_free(evcon);
        delete request;
        return false;
    }

    // keep the external callback function object on the heap, so it is allowed
    // to run out of scope on the calling side
    // also pass the connection to the callback, so it can be freed there
    std::pair<std::function<void(restReply)>, struct evhttp_connection*>* pair =
        new std::pair<std::function<void(restReply)>, struct evhttp_connection*>(
            request->request_done_cb, evcon);
    req = evhttp_request_new(http_request_done, pair);
    if (req == nullptr) {
        WARN_NON_OO("restClient: evhttp_request_new() failed.");
        evhttp_connection_free(evcon);
        delete request;
        return false;
    }
    output_headers = evhttp_request_get_output_headers(req);
    ret = evhttp_add_header(output_headers, "Host", request->host->c_str());
    if (ret) {
        WARN_NON_OO("restClient: Failure adding \"Host\" header.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        delete request;
        return false;
    }
    ret = evhttp_add_header(output_headers, "Connection", "close");
    if (ret) {
        WARN_NON_OO("restClient: Failure adding \"Connection\" header.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        delete request;
        return false;
    }
    ret = evhttp_add_header(output_headers, "Content-Type", "application/json");
    if (ret) {
        WARN_NON_OO("restClient: Failure adding \"Content-Type\" header.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        delete request;
        return false;
    }
    if (!request->data->empty()) {
        size_t datalen = request->data->dump().length();
        char buf[256];

        struct evbuffer* output_buffer = evhttp_request_get_output_buffer(req);

        // copy data into the buffer
        ret = evbuffer_add(output_buffer, request->data->dump().c_str(), datalen);
        if (ret) {
            WARN_NON_OO("restClient: Failure adding JSON data to event buffer.");
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            delete request;
            return false;
        }

        evutil_snprintf(buf, sizeof(buf) - 1, "%lu", (unsigned long)datalen);
        ret = evhttp_add_header(output_headers, "Content-Length", buf);
        if (ret) {
            WARN_NON_OO("restClient: Failure adding \"Content-Length\" header.");
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            delete request;
            return false;
        }
        DEBUG2_NON_OO("restClient: Sending {:s} bytes: {:s}", buf, request->data->dump(4));

        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_POST, request->path->c_str());
    } else {
        DEBUG2_NON_OO("restClient: sending GET request.");
        ret = evhttp_make_request(evcon, req, EVHTTP_REQ_GET, request->path->c_str());
    }
    if (ret) {
        WARN_NON_OO("restClient: evhttp_make_request() failed.");
        evhttp_connection_free(evcon);
        evhttp_request_free(req);
        delete request;
        return false;
    }
    delete request;
    return true;
}

restReply restClient::make_request_blocking(const std::string& path, const nlohmann::json& data,
                                            const std::string& host, const unsigned short port,
                                            const int retries, const int timeout) {
    restReply reply = restReply(false, "");
    bool reply_copied = false;

    // Condition variable to signal that reply was copied
    std::condition_variable cv_reply;
    std::mutex mtx_reply;

    // As a callback, pass a lambda that synchronizes copying the reply in here.
    std::function<void(restReply)> callback([&](restReply reply_in) {
        std::lock_guard<std::mutex> lck_reply(mtx_reply);
        reply = reply_in;
        reply_copied = true;
        cv_reply.notify_one();
    });

    std::unique_lock<std::mutex> lck_reply(mtx_reply);

    make_request(path, callback, data, host, port, retries, timeout);

    // Wait for the callback to receive the reply.
    // Note: This timeout is only in case libevent for any reason never
    // calls the callback lambda we pass to it. That's a serious error case.
    // In a normal timeout situation, we have to make sure libevent times out
    // before this, that's why we wait twice as long.
    auto time_point =
        std::chrono::system_clock::now() + std::chrono::seconds(timeout == -1 ? 100 : timeout * 2);
    while (!cv_reply.wait_until(lck_reply, time_point, [&]() { return reply_copied; })) {
        FATAL_ERROR_NON_OO("restClient: Timeout in make_request_blocking ({:s}:{:d}/{:s}). This "
                           "might leave the restClient in an abnormal state. Exiting...",
                           host, port, path);
        return reply;
    }
    return reply;
}

void restClient::_bev_req_errcb(bufferevent* bev, short what, void* arg) {
    (void)arg;
    std::string err = "";

    // find out what the error code means
    if (what & BEV_EVENT_READING)
        err += "error encountered while reading, ";
    if (what & BEV_EVENT_WRITING)
        err += "error encountered while writing, ";
    if (what & BEV_EVENT_EOF)
        err += "eof file reached, ";
    if (what & BEV_EVENT_ERROR)
        err += "unrecoverable error encountered, ";
    if (what & BEV_EVENT_TIMEOUT)
        err += "user-specified timeout reached, ";
    if (what & BEV_EVENT_CONNECTED)
        err += "connect operation finished.";

    WARN_NON_OO("restClient::_bev_req_errcb: {:p} - {:d} ({:s})", (void*)bev, what, err);
}
