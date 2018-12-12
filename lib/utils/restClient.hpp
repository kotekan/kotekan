/*****************************************
@file
@brief Send REST requests to a server.
*****************************************/
#ifndef RESTCLIENT_HPP
#define RESTCLIENT_HPP

#include <event2/util.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "json.hpp"

#include "Config.hpp"
#include "restServer.hpp"

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
    static restClient &instance();

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
     * @return          `true` if successfull, otherwise `false`.
     */
    bool make_request(const std::string &path,
                      std::function<void(restReply)> request_done_cb,
                      const nlohmann::json& data = {},
                      const std::string& host = "127.0.0.1",
                      const unsigned short port = PORT_REST_SERVER,
                      const int retries = 0, const int timeout = -1);

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
    restReply make_request_blocking(
            const std::string& path,
            const nlohmann::json& data = {},
            const std::string& host = "127.0.0.1",
            const unsigned short port = PORT_REST_SERVER,
            const int retries = 0, const int timeout = -1);

private:
    /// Private constuctor
    restClient();

    /// Private destructor
    virtual ~restClient();

    /// @brief Internal timer call back to check for thread exit condition
    static void timer(evutil_socket_t fd, short event, void *arg);

    /// Do not allow copy or assignment
    restClient(restClient const&);
    void operator=(restClient const&);

    /// Internal thread function which runs the event loop.
    void event_thread();

    /// callback function for http requests
    static void http_request_done(struct evhttp_request *req, void *arg);

    /// cleanup function that deletes evcon and the argument pair
    static void cleanup(std::pair<std::function<void(restReply)>,
                                         struct evhttp_connection*>* pair);

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
};

#endif // RESTCLIENT_HPP

