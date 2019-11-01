/*****************************************
@file
@brief Send REST requests to a server.
*****************************************/
#ifndef RESTCLIENT_HPP
#define RESTCLIENT_HPP

#include "Config.hpp"
#include "restServer.hpp"

#include "json.hpp"

#include <atomic>
#include <condition_variable>
#include <event2/util.h>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

// The reply of a request: a pair with a success boolean and the reply string
using restReply = std::pair<bool, std::string>;

/**
 * @class restClient
 * @brief REST client: Send REST messages to a server and maybe get a reply.
 *
 * This class supports sending GET messages and POST messages with json data
 * using libevent and provides access to data from the reply of the server.
 *
 * @author Rick Nitsche
 */
class restClient {
public:
    /**
     * @brief Returns an instance of the rest client.
     *
     * @return Returns the rest client instance.
     */
    static restClient& instance();

    /**
     * @brief Send GET or POST with json data to an endpoint.
     *
     * To send a GET message, pass an empty JSON object (`{}`) as the parameter
     * `data`. To send a POST message, pass JSON data.
     *
     * @param path      Path to the endpoint
     *                  (e.g. "/endpoint_name")
     * @param request_done_cb   A callback function that when the request is
     *                          complete.
     * @param data      JSON request (`{}` to send a GET request,
     *                  default: `{}`).
     * @param host      Host (default: "127.0.0.1", Prefer numerical, because
     *                  the DNS lookup is blocking).
     * @param port      Port (default: PORT_REST_SERVER).
     * @param retries   Max. retries to send message (default: 0).
     * @param timeout   Timeout in seconds. If -1 is passed, the default value
     * (of 50 seconds) is set (default: -1).
     */
    void make_request(const std::string& path, std::function<void(restReply)> request_done_cb,
                      const nlohmann::json& data = {}, const std::string& host = "127.0.0.1",
                      const unsigned short port = PORT_REST_SERVER, const int retries = 0,
                      const int timeout = -1);

    /**
     * @brief Send GET or POST with json data to an endpoint. Blocking.
     *
     * To send a GET message, pass an empty JSON object (`{}`) as the parameter
     * `data`. To send a POST message, pass JSON data. This blocks until a reply
     * is received or there was an error.
     *
     * @param path      Path to the endpoint
     *                  (e.g. "/endpoint_name")
     * @param data      JSON request (`{}` to send a GET request,
     *                  default: `{}`).
     * @param host      Host (default: "127.0.0.1", Prefer numerical, because
     *                  the DNS lookup is blocking).
     * @param port      Port (default: PORT_REST_SERVER).
     * @param retries   Max. retries to send message (default: 0).
     * @param timeout   Timeout in seconds. If -1 is passed, the default value
     * (of 50 seconds) is set (default: -1).
     *
     * @return          restReply object.
     */
    restReply make_request_blocking(const std::string& path, const nlohmann::json& data = {},
                                    const std::string& host = "127.0.0.1",
                                    const unsigned short port = PORT_REST_SERVER,
                                    const int retries = 0, const int timeout = -1);

private:
    /// A structure to pass requests around inside the restClient
    struct restRequest {
        std::string const* path;
        std::function<void(restReply)> request_done_cb;
        nlohmann::json const* data;
        std::string const* host;
        unsigned short port;
        int retries;
        int timeout;

        restRequest(const std::string& path, std::function<void(restReply)> request_done_cb,
                    const nlohmann::json& data, const std::string& host, const unsigned short port,
                    const int retries, const int timeout) :
            path(new std::string(path)),
            request_done_cb(request_done_cb),
            data(new nlohmann::json(data)),
            host(new std::string(host)),
            port(port),
            retries(retries),
            timeout(timeout) {}

        ~restRequest() {
            delete path;
            delete data;
            delete host;
        }
    };

    /// Private constuctor
    restClient();

    /// Private destructor
    virtual ~restClient();

    /// @brief Internal timer call back to check for thread exit condition
    static void timer(evutil_socket_t fd, short event, void* arg);

    /// Do not allow copy or assignment
    restClient(restClient const&);
    void operator=(restClient const&);

    /// Internal thread function which runs the event loop.
    void event_thread();

    /// callback function for http requests
    static void http_request_done(struct evhttp_request* req, void* arg);

    /// cleanup function that deletes evcon and the argument pair
    static void cleanup(std::pair<std::function<void(restReply)>, struct evhttp_connection*>* pair);

    /// Only to be called inside event thread
    bool _make_request(const restRequest* request);

    /// Read callback for the request scheduling thread. Called when the socket holding request
    /// pointers has new data.
    static void _bev_req_readcb(struct bufferevent* bev, void* arg);

    /// error callback for the bufferevent pair
    static void _bev_req_errcb(struct bufferevent* bev, short what, void* arg);

    /// Main event thread handle
    std::thread _main_thread;

    /// flag set to true when exit condition is reached
    std::atomic<bool> _stop_thread;

    /// event base
    /// TODO: use the event base of the restServer
    struct event_base* _base;

    /// dns base
    struct evdns_base* _dns;

    /// Condition variable to signal that event thread has started
    std::condition_variable _cv_start;
    bool _event_thread_started;
    std::mutex _mtx_start;

    /// Writing socket event to pass requests to the event thread
    struct bufferevent* bev_req_write;

    /// Reading socket event to pass requests to the event thread
    struct bufferevent* bev_req_read;
};

#endif // RESTCLIENT_HPP
