#include "restClient.hpp"

#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO, DEBUG_NON_OO, WARN_NON_OO

#include <chrono>                      // for operator+, seconds, system_clock, system_clock::t...
#include <condition_variable>          // for condition_variable
#include <cstring>                     // for memcpy
#include <event2/buffer.h>             // for iovec, evbuffer_peek, evbuffer_iovec, evbuffer
#include <event2/bufferevent.h>        // for bufferevent_read, bufferevent_free, bufferevent_s...
#include <event2/bufferevent_struct.h> // for bufferevent
#include <event2/dns.h>                // for evdns_base_free, evdns_base_new
#include <event2/event.h>              // for event_base_loopbreak, EV_READ, event_add, event_b...
#include <event2/http.h>               // for evhttp_connection_free, evhttp_request_free, evht...
#include <event2/keyvalq_struct.h>     // for evkeyvalq
#include <event2/thread.h>             // for evthread_use_pthreads
#include <evhttp.h>                    // for evhttp_request
#include <pthread.h>                   // for pthread_setname_np
#include <stdlib.h>                    // for free, malloc
#include <sys/time.h>                  // for timeval
#include <vector>                      // for __alloc_traits<>::value_type


restClient& restClient::instance() {
    static restClient client_instance;
    return client_instance;
}

restClient::restClient() :
    _main_thread() {

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
    // Free the various libevent objects
    if (bev_req_write)
        bufferevent_free(bev_req_write);
    if (bev_req_read)
        bufferevent_free(bev_req_read);
    if (timer_event)
        event_free(timer_event);
    if (_dns)
        evdns_base_free(_dns, 1);
    if (_base)
        event_base_free(_base);
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
        return;
    }

    // Create the base event, and exclude using `select` as a backend API
    event_config* ev_config = event_config_new();
    if (!ev_config) {
        FATAL_ERROR_NON_OO("Failed to create config for libevent");
        return;
    }
    int err = event_config_avoid_method(ev_config, "select");
    if (err) {
        FATAL_ERROR_NON_OO("Failed to exclude select from the libevent options");
        return;
    }
    _base = event_base_new_with_config(ev_config);
    if (!_base) {
        FATAL_ERROR_NON_OO("restClient: Failure creating new event_base.");
        return;
    }

    // The event loop will run in this seperate thread. We have to schedule requests from
    // this same thread. Therefor we create a bufferevent pair here to pass request pointers in.
    bufferevent* pair[2];
    if (bufferevent_pair_new(_base, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE, pair)) {
        FATAL_ERROR_NON_OO("restClient: Failure creating bufferevent pair.");
        return;
    }
    bev_req_write = pair[0];
    bev_req_read = pair[1];
    if (bufferevent_disable(bev_req_write, EV_READ)) {
        FATAL_ERROR_NON_OO("restClient: Failure in bufferevent_disable().");
        return;
    }
    if (bufferevent_enable(bev_req_read, EV_READ)) {
        FATAL_ERROR_NON_OO("restClient: Failure in bufferevent_enable().");
        return;
    }

    // Set a watermark on the input buffer of the bufferevent for reading. This is to prevent
    // it getting filled up so much, that the read callback later starves other threads.
    bufferevent_setwatermark(bev_req_read, EV_READ, 0, 5 * 1 << 20); // 5mb

    bufferevent_setcb(bev_req_read, _bev_req_readcb, nullptr, _bev_req_errcb, this);
    bufferevent_setcb(bev_req_write, nullptr, nullptr, _bev_req_errcb, this);

    // DNS resolution is blocking (if not numeric host is passed)
    _dns = evdns_base_new(_base, 1);
    if (_dns == nullptr) {
        FATAL_ERROR_NON_OO("restClient: evdns_base_new() failed.");
        return;
    }

    // Create a timer to check for the exit condition
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
            break;
        }
    }
    DEBUG_NON_OO("restClient: exiting event loop");
}

void restClient::http_request_done(struct evhttp_request* req, void* arg) {
    // FIXME: evcon is passed here, because evhttp_request_get_connection(req)
    // always returns a nullptr and there is no way to free the connection on
    // completion in libevent < 2.1
    // libevent 2.1 has a evhttp_connection_free_on_completion, but using it,
    // the bufferevent never gets deleted...
    // TODO: maybe keep the evhttp_connections in a pool and reuse them
    // (set Connection:keep-alive header)
    auto pair = (std::pair<std::function<void(restReply)>*, struct evhttp_connection*>*)arg;
    std::function<void(restReply)>* ext_cb = pair->first;

    // this is where we store the reply
    std::string str_data("");

    if (req == nullptr) {
        int errcode = EVUTIL_SOCKET_ERROR();
        // Print socket error
        std::string str = evutil_socket_error_to_string(errcode);
        WARN_NON_OO("restClient: Request failed with socket error {:d} ({:s})", errcode, str);
        (*ext_cb)(restReply(false, str_data));
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
        (*ext_cb)(restReply(false, str_data));
        cleanup(pair);
        return;
    }

    // get input buffer
    evbuffer* input_buffer = evhttp_request_get_input_buffer(req);
    size_t datalen = evbuffer_get_length(input_buffer);
    if (datalen == 0) {
        (*ext_cb)(restReply(true, str_data));
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
    int n_vec = evbuffer_peek(input_buffer, datalen, nullptr, nullptr, 0);
    if (n_vec < 0) {
        WARN_NON_OO("restClient: Failure in evbuffer_peek()");
        (*ext_cb)(restReply(false, str_data));
        cleanup(pair);
        return;
    }

    // Allocate space for the chunks.
    vec_out = (evbuffer_iovec*)malloc(sizeof(evbuffer_iovec) * n_vec);

    n_vec = evbuffer_peek(input_buffer, datalen, nullptr, vec_out, n_vec);
    for (int i = 0; i < n_vec; i++) {
        size_t len = vec_out[i].iov_len;
        if (written + len > datalen)
            len = datalen - written;
        str_data.append((char*)vec_out[i].iov_base, len);
        written += len;
    }
    free(vec_out);

    // call the external callback
    (*ext_cb)(restReply(true, str_data));
    cleanup(pair);
}

void restClient::cleanup(
    std::pair<std::function<void(restReply)>*, struct evhttp_connection*>* pair) {
    if (pair->second)
        evhttp_connection_free(pair->second);
    delete pair;
}

void restClient::make_request(const std::string& path,
                              std::function<void(restReply)>& request_done_cb,
                              const nlohmann::json& data, const std::string& host,
                              const unsigned short port, const int retries, const int timeout) {
    DEBUG2_NON_OO("restClient::make_request(): {}:{}{}, data = {}", host, port, path, data.dump(4));

    if (!bev_req_write || !bev_req_read)
        FATAL_ERROR_NON_OO(
            "restClient: make_request called, but bev_req_write returned a nullptr.");

    // serialize json data
    std::string datadump = data.dump();

    // put as much of the request data as possible in a struct
    restRequest request = {
        datadump.size(), path.size(), host.size(), retries, timeout, port, &request_done_cb,
    };
    if (data.empty())
        request.data_len = 0;

    // We have a struct and 3 strings to drop into the bufferevent.
    // We only want the read callback to be called once we are done. That's why we use
    // evbuffer_iovec to copy all our data in before committing that to the bufferevent.
    evbuffer_iovec iovec[2];
    size_t len_total = sizeof(restRequest) + request.host_len + request.path_len + request.data_len;
    evbuffer* output_buf = bufferevent_get_output(bev_req_write);

    // After we reserve space we have to make sure no other thread writes to the buffer before
    // we commit. Therefore lock this critical section with a mutex.
    std::lock_guard<std::mutex> lock_bev_buffer(_mtx_bev_buffer);
    int n_extends = evbuffer_reserve_space(output_buf, len_total, iovec, 2);
    if (n_extends < 0)
        FATAL_ERROR_NON_OO(
            "restClient::make_request: unable to reserve memory (evbuffer_reserve_space).");

    int i_extends = 0;
    size_t i_vec = 0;
    _copy_to_iovec(&request, sizeof(request), iovec, &i_extends, &i_vec, n_extends);
    _copy_to_iovec(host.c_str(), request.host_len, iovec, &i_extends, &i_vec, n_extends);
    _copy_to_iovec(path.c_str(), request.path_len, iovec, &i_extends, &i_vec, n_extends);
    if (request.data_len)
        _copy_to_iovec(datadump.c_str(), request.data_len, iovec, &i_extends, &i_vec, n_extends);

    // make sure we don't commit too much
    iovec[i_extends].iov_len = i_vec;

    if (evbuffer_commit_space(output_buf, iovec, i_extends + 1) < 0)
        FATAL_ERROR_NON_OO("restClient::make_request: Failure in evbuffer_commit_space.");
}

void restClient::_copy_to_iovec(const void* src, size_t len_src, evbuffer_iovec* iovec,
                                int* i_extends, size_t* i_vec, int n_extends) {
    // If we have more data than fits in the iovec, copy as much as possible and continue
    // with the next extension of the iovec.
    while (len_src > iovec[*i_extends].iov_len) {
        std::memcpy(static_cast<char*>(iovec[*i_extends].iov_base) + *i_vec, src,
                    iovec[*i_extends].iov_len);
        src = static_cast<const char*>(src) + iovec[*i_extends].iov_len;
        len_src -= iovec[*i_extends].iov_len;
        *i_vec = 0;
        if (++(*i_extends) >= n_extends)
            FATAL_ERROR_NON_OO(
                "restClient::_copy_to_iovec: not enough iovec extends to copy data.");
    }

    // If the iovec has enough space, copy all and be done.
    std::memcpy(static_cast<char*>(iovec[*i_extends].iov_base) + *i_vec, src, len_src);
    *i_vec += len_src;
    return;
}

void restClient::_bev_req_readcb(struct bufferevent* bev, void* arg) {
    // Get the restClient object from the arg void pointer
    restClient* client = (restClient*)arg;
    if (!client)
        FATAL_ERROR_NON_OO("restClient: _bev_req_readcb got nullptr client");

    // libevent calls this cb just once when there are multiple requests in the
    // bufferevent pair. Read until we have all of them.
    size_t len_read;
    while (true) {
        restRequest request;
        len_read = 0;
        do {
            len_read += bufferevent_read(bev, &request + len_read, sizeof(restRequest) - len_read);
            // this can be empty ...means there's nothing to do.
            if (len_read == 0)
                return;
        } while (len_read < sizeof(restRequest));
        DEBUG2_NON_OO("restClient: _bev_req_readcb read {} bytes", len_read);
        if (len_read != sizeof(restRequest))
            FATAL_ERROR_NON_OO("restClient::_bev_req_readcb: Failure reading request struct from "
                               "bufferevent ({} != {}).",
                               len_read, sizeof(restRequest));

        std::string host;
        host.resize(request.host_len);
        len_read = 0;
        do
            len_read += bufferevent_read(bev, &host[0] + len_read, request.host_len - len_read);
        while (len_read < request.host_len);
        if (len_read != request.host_len)
            FATAL_ERROR_NON_OO("restClient::_bev_req_readcb: Failure reading hostname from "
                               "bufferevent ({} != {}).",
                               len_read, request.host_len);

        std::string path;
        path.resize(request.path_len);
        len_read = 0;
        do
            len_read += bufferevent_read(bev, &path[0] + len_read, request.path_len - len_read);
        while (len_read < request.path_len);
        if (len_read != request.path_len)
            FATAL_ERROR_NON_OO("restClient::_bev_req_readcb: Failure reading endpoint from "
                               "bufferevent ({} != {}).",
                               len_read, request.path_len);

        evhttp_connection* evcon =
            evhttp_connection_base_new(client->_base, client->_dns, host.c_str(), request.port);
        if (evcon == nullptr)
            FATAL_ERROR_NON_OO("restClient: evhttp_connection_base_new() failed.");

        if (request.retries > 0)
            evhttp_connection_set_retries(evcon, request.retries);
        if (request.timeout >= 0)
            evhttp_connection_set_timeout(evcon, request.timeout);

        // Fire off the request and pass the external callback to the internal one.
        // check if external callback function is callable
        if (!request.request_done_cb) {
            evhttp_connection_free(evcon);
            FATAL_ERROR_NON_OO("restClient: external callback function is not callable.");
        }

        // pass the external callback function to the libevent callback
        // also pass the connection to the callback, so it can be freed there
        std::pair<std::function<void(restReply)>*, evhttp_connection*>* pair =
            new std::pair<std::function<void(restReply)>*, evhttp_connection*>(
                request.request_done_cb, evcon);
        evhttp_request* req = evhttp_request_new(http_request_done, pair);
        if (req == nullptr) {
            evhttp_connection_free(evcon);
            FATAL_ERROR_NON_OO("restClient: evhttp_request_new() failed.");
        }
        evkeyvalq* output_headers = evhttp_request_get_output_headers(req);
        if (evhttp_add_header(output_headers, "Host", host.c_str())) {
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            FATAL_ERROR_NON_OO("restClient: Failure adding \"Host\" header.");
        }
        if (evhttp_add_header(output_headers, "Connection", "close")) {
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            FATAL_ERROR_NON_OO("restClient: Failure adding \"Connection\" header.");
        }
        if (evhttp_add_header(output_headers, "Content-Type", "application/json")) {
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            FATAL_ERROR_NON_OO("restClient: Failure adding \"Content-Type\" header.");
        }

        int ret;
        if (request.data_len) {
            char buf[256];

            evbuffer* output_buffer = evhttp_request_get_output_buffer(req);
            evbuffer* bev_buffer = bufferevent_get_input(bev);

            // move data into the requests output buffer (avoids copy by value)
            len_read = 0;
            do
                len_read +=
                    evbuffer_remove_buffer(bev_buffer, output_buffer, request.data_len - len_read);
            while (len_read < request.data_len);
            if (len_read != request.data_len) {
                evhttp_connection_free(evcon);
                evhttp_request_free(req);
                FATAL_ERROR_NON_OO("restClient::_bev_req_readcb: Failure reading data from "
                                   "bufferevent ({} != {}).",
                                   len_read, request.data_len);
            }

            evutil_snprintf(buf, sizeof(buf) - 1, "%lu", (unsigned long)request.data_len);
            if (evhttp_add_header(output_headers, "Content-Length", buf)) {
                evhttp_connection_free(evcon);
                evhttp_request_free(req);
                FATAL_ERROR_NON_OO("restClient: Failure adding \"Content-Length\" header.");
            }
            DEBUG_NON_OO("restClient: Sending {:s} bytes.", buf);

            ret = evhttp_make_request(evcon, req, EVHTTP_REQ_POST, path.c_str());
        } else {
            DEBUG_NON_OO("restClient: sending GET request.");
            ret = evhttp_make_request(evcon, req, EVHTTP_REQ_GET, path.c_str());
        }
        if (ret) {
            evhttp_connection_free(evcon);
            evhttp_request_free(req);
            FATAL_ERROR_NON_OO("restClient: evhttp_make_request() failed.");
        }
    }
}

restClient::restReply restClient::make_request_blocking(const std::string& path,
                                                        const nlohmann::json& data,
                                                        const std::string& host,
                                                        const unsigned short port,
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
